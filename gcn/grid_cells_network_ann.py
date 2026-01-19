import numpy as np
import torch
from my_func.func import cann3d_pre_encoder,get_delta_gcn

class CANN3Dnet(torch.nn.Module):
    def __init__(self , nums):
        super(CANN3Dnet, self).__init__()
        self.nums = nums
        tri_nums= nums * 3
        self.tri_nums = tri_nums
        self.input_net = torch.nn.Sequential(torch.nn.Linear(tri_nums, tri_nums, bias=False),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(tri_nums, tri_nums, bias=False))
        self.state_net = torch.nn.LSTM(input_size = tri_nums, hidden_size = tri_nums, num_layers=2, bias = False, batch_first=True)

        self.decoder_net = torch.nn.Sequential(torch.nn.Linear(tri_nums, nums),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(nums, 6),
                                               )

    def forward(self, x, hn ,cn):
        input = self.input_net(x)
        state, (hn, cn) = self.state_net(input, (hn, cn))
        decoder_output = self.decoder_net(state)
        # ( batch_size , dims)
        return [input, state, decoder_output], (hn, cn)

class GridCellNetwork:
    def __init__(self, nums, weight_path, **kwargs):
        self.cann_ann = CANN3Dnet(nums)
        self.cann_ann.load_state_dict(torch.load(weight_path))
        self.cann_ann.eval()
        self.GC_X_DIM = nums - 1
        self.GC_Y_DIM = nums - 1
        self.GC_Z_DIM = nums - 1
        self.GC_X_RANGE = kwargs.pop("GC_X_RANGE", nums-1)
        self.GC_Y_RANGE = kwargs.pop("GC_Y_RANGE", nums-1)
        self.GC_Z_RANGE = kwargs.pop("GC_Z_RANGE", nums-1)
        self.GC_X_RES = self.GC_X_RANGE / self.GC_X_DIM
        self.GC_Y_RES = self.GC_Y_RANGE / self.GC_X_DIM
        self.GC_Z_RES = self.GC_Z_RANGE / self.GC_X_DIM
        self.center_init = kwargs.pop("center_init", True)
        #19 nueros only have 18 intervals for the identity between the first one and the last one
        self.GC_X_TH_SIZE = np.pi / (self.GC_X_DIM // 2)
        self.GC_Y_TH_SIZE = np.pi / (self.GC_X_DIM // 2)
        self.GC_Z_TH_SIZE = np.pi / (self.GC_X_DIM // 2)
        self.GC_RES = torch.tensor((self.GC_X_RES, self.GC_Y_RES, self.GC_Z_RES)).reshape(1, 3)
        self.GC_TH_SIZE = torch.tensor((self.GC_X_TH_SIZE, self.GC_Y_TH_SIZE, self.GC_Z_TH_SIZE)).reshape(1, 3)
        #cann2ann
        #initailize location and hidden statae
        self.hidden_state = (torch.zeros(2, self.cann_ann.tri_nums), torch.zeros(2, self.cann_ann.tri_nums))
        gcX, gcY, gcZ = self.get_gc_initial_pos()
        self.MAX_ACTIVE_XYZ_PATH = [gcX, gcY, gcZ]
        self.pre_location = torch.tensor(self.MAX_ACTIVE_XYZ_PATH).reshape(1,3)
        #recording list
        self.cann_ann_history = []

    def get_gc_initial_pos(self):
        if self.center_init:
            gcX = self.GC_X_DIM // 2
            gcY = self.GC_Y_DIM // 2
            gcZ = self.GC_Z_DIM // 2
            return gcX, gcY, gcZ
        return 0 , 0 , 0

    def get_gc_xyz(self,cann_state):
        pre_decoded_output = np.array(self.cann_ann.decoder_net(cann_state))
        #get the output in radian
        gc_x = np.arctan2(pre_decoded_output[0, 0], pre_decoded_output[0, 1])
        gc_y = np.arctan2(pre_decoded_output[0, 2], pre_decoded_output[0, 3])
        gc_z = np.arctan2(pre_decoded_output[0, 4], pre_decoded_output[0, 5])
        #from radian to idx
        gc_x = np.mod(gc_x + np.pi, 2 * np.pi) / self.GC_X_TH_SIZE
        gc_y = np.mod(gc_y + np.pi, 2 * np.pi) / self.GC_Y_TH_SIZE
        gc_z = np.mod(gc_z + np.pi, 2 * np.pi) / self.GC_Z_TH_SIZE
        return gc_x,gc_y,gc_z

    def gc_iteration(self, vt_id, transV, curYawThetaInRadian, heightV, VT):
        """
        updatae_steps:
            1.get the location which needs to be encoded
            2.ANN forward
            3.decoding state
        """
        delta_x = -transV * np.sin(curYawThetaInRadian)
        delta_y = transV * np.cos(curYawThetaInRadian)
        delta_z = heightV
        cur_location = self.pre_location + torch.tensor((delta_x,delta_y,delta_z)).reshape(1,3)/ self.GC_RES
        self.pre_location = cur_location
        cur_location_encoded = cur_location * self.GC_TH_SIZE - torch.pi
        cur_location_encoded = cann3d_pre_encoder(cur_location_encoded, self.cann_ann.nums, concat=True)
        with torch.no_grad():
            GC_input = self.cann_ann.input_net(cur_location_encoded)
            cann_state, self.hidden_state = self.cann_ann.state_net(GC_input, self.hidden_state)
            # self.cann_ann_history.append(cann_state.reshape(111))
            self.MAX_ACTIVE_XYZ_PATH = list(self.get_gc_xyz(cann_state))
        if VT[vt_id].first != 1:
            #获得VT记录的索引
            actX = np.mod(VT[vt_id].gc_x, self.GC_X_DIM)
            actY = np.mod(VT[vt_id].gc_y, self.GC_Y_DIM)
            actZ = np.mod(VT[vt_id].gc_z, self.GC_Z_DIM)
            bais = get_delta_gcn(self.MAX_ACTIVE_XYZ_PATH, [actX, actY, actZ], self.GC_X_DIM)
            self.MAX_ACTIVE_XYZ_PATH = np.mod(np.array(self.MAX_ACTIVE_XYZ_PATH) + bais, self.GC_X_DIM).tolist()
            self.pre_location += torch.tensor(bais)
        return self.MAX_ACTIVE_XYZ_PATH
