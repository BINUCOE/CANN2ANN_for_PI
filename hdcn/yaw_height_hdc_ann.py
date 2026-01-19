import numpy as np
import torch
from my_func.func import pre_encoder, get_delta_hdc
class CANNnet(torch.nn.Module):
    def __init__(self,nums):
        super(CANNnet , self).__init__()
        self.nums =nums
        self.input_net = torch.nn.Sequential(torch.nn.Linear(nums, nums, bias=False),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(nums, nums ,bias=False),
                                             )
        self.state_net = torch.nn.LSTM(input_size=nums, hidden_size=nums, num_layers=2, bias=False)

        self.decoder_net = torch.nn.Sequential(torch.nn.Linear(nums, nums // 3),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(nums // 3, 2),
                                               )
    def forward(self , x, hn = None, cn = None):
        input = self.input_net(x)
        state,(hn,cn) = self.state_net(input,(hn,cn))
        decoder_output = self.decoder_net(state)
        #( batch_size , dims)
        return [input, state, decoder_output],(hn, cn)

class Yaw_Height_HDC_ANN:
    def __init__(self, nums, weight_path , **kwargs):
        self.cann_ann = CANNnet(nums)
        self.cann_ann.load_state_dict(torch.load(weight_path))
        self.cann_ann.eval()
        half_num = nums // 2
        #确定编码的高度范围
        self.YAW_HEIGHT_HDC_Y_DIM = nums - 1
        self.YAW_HEIGHT_HDC_Y_RANGE = kwargs.pop('YAW_HEIGHT_HDC_Y_RANGE', 2 * np.pi)
        # The dimension and range of height in yaw_height_hdc network
        self.YAW_HEIGHT_HDC_H_DIM = nums - 1
        self.YAW_HEIGHT_HDC_H_RANGE = kwargs.pop('YAW_HEIGHT_HDC_H_RANGE', self.YAW_HEIGHT_HDC_H_DIM)
        self.center_init = kwargs.pop("center_init", False)
        self.YAW_HEIGHT_HDC_Y_TH_SIZE = np.pi / half_num
        self.YAW_HEIGHT_HDC_H_SIZE = np.pi / half_num
        self.YAW_HEIGHT_HDC_Y_RES = self.YAW_HEIGHT_HDC_Y_RANGE / self.YAW_HEIGHT_HDC_Y_DIM
        self.YAW_HEIGHT_HDC_H_RES = self.YAW_HEIGHT_HDC_H_RANGE / self.YAW_HEIGHT_HDC_H_DIM
        #通过设置累加器解决高度信息非周期性的问题
        self.height_buffer = 0
        #LSTM隐藏状态维护
        self.height_hidden_state = (torch.zeros(2, nums), torch.zeros(2,  nums))
        self.yaw_hidden_state = (torch.zeros(2, nums), torch.zeros(2,  nums))
        self.YAW_HEIGHT_HDC_VT_INJECT_ENERGY = 0.1
        # self.yaw_state_history = []
        # self.height_state_history = []
        curYawTheta, curHeight = self.get_hdc_initial_value()
        self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = [curYawTheta, curHeight]
        self.target = np.array(self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH,dtype=np.float64)

    def get_hdc_initial_value(self):
        if self.center_init:
            curYawTheta = self.YAW_HEIGHT_HDC_Y_DIM // 2
            curHeight = self.YAW_HEIGHT_HDC_H_DIM // 2
        else:
            curYawTheta = 0
            curHeight = 0
        return curYawTheta, curHeight

    def get_current_yaw_height_value(self ,yaw_cann_state,height_cann_state,heightV):
        #返回解码出的角度值,高度值
        yaw_decoder_output = np.array(self.cann_ann.decoder_net(yaw_cann_state))
        yaw_decoder_output = np.arctan2(yaw_decoder_output[0, 0] , yaw_decoder_output[0, 1])
        yaw_decoder_output = np.mod(yaw_decoder_output + np.pi, 2 * np.pi) / self.YAW_HEIGHT_HDC_Y_TH_SIZE
        # self.yaw_decoded_history.append(yaw_decoder_output)
        if heightV == 0:
            height_decoder_output = self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH[1]
        else:
            height_decoder_output = np.array(self.cann_ann.decoder_net(height_cann_state))
            height_decoder_output = np.arctan2(height_decoder_output[0, 0] , height_decoder_output[0, 1])
            height_decoder_output = np.mod(height_decoder_output + np.pi , 2 * np.pi) / self.YAW_HEIGHT_HDC_H_SIZE
        return yaw_decoder_output, height_decoder_output

    def yaw_height_hdc_iteration(self, vt_id, yawRotV, heightV, VT):
        #目标编码高度和目标编码角度计算，类似imu的累进形式
        self.target[0] = self.target[0] + yawRotV / self.YAW_HEIGHT_HDC_Y_RES
        self.target[1] = self.target[1] + heightV / self.YAW_HEIGHT_HDC_H_RES
        cur_target_yaw = self.target[0] * self.YAW_HEIGHT_HDC_Y_TH_SIZE - torch.pi
        # self.target_yaw_history.append(cur_target_yaw)
        cur_target_height = self.target[1] * self.YAW_HEIGHT_HDC_H_SIZE - torch.pi
        cur_target_yaw_encoded = pre_encoder(torch.tensor(cur_target_yaw).reshape(1, 1),self.cann_ann.nums)
        cur_target_height_encoded = pre_encoder(torch.tensor(cur_target_height).reshape(1 , 1),self.cann_ann.nums)
        with torch.no_grad():
            yaw_input = self.cann_ann.input_net(cur_target_yaw_encoded)
            yaw_cann_state, self.yaw_hidden_state = self.cann_ann.state_net(yaw_input, self.yaw_hidden_state)
            # self.yaw_state_history.append(yaw_cann_state.reshape(37))
            if heightV == 0:
                height_cann_state = None
                # if len(self.height_state_history) == 0:
                #     self.height_state_history.append(np.zeros(37))
                # else:
                #     self.height_state_history.append(self.height_state_history[-1])
            else:
                height_input = self.cann_ann.input_net(cur_target_height_encoded)
                height_cann_state , self.height_hidden_state = self.cann_ann.state_net(height_input, self.height_hidden_state)
                # self.height_state_history.append(height_cann_state.reshape(37))
            self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = list(self.get_current_yaw_height_value(yaw_cann_state , height_cann_state, heightV))
            if VT[vt_id].first != 1:
                act_yaw = VT[vt_id].hdc_yaw
                act_height = VT[vt_id].hdc_height
                #根据当前值与VT值计算所需产生的偏移修正
                bais = get_delta_hdc(self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH, [act_yaw, act_height], self.YAW_HEIGHT_HDC_H_DIM)
                #修正输出
                self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = np.mod(np.array(self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH) + bais,
                                                             self.YAW_HEIGHT_HDC_Y_DIM).tolist()
                self.target += bais
        return self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH

