import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras import Model,layers
from my_func.func_keras import pre_encoder_keras ,get_delta_hdc

class CANN1DNet(Model):
    def __init__(self,nums):
        super(CANN1DNet, self).__init__()
        self.nums = nums
        self.input_net = tf.keras.Sequential([
            layers.Dense(nums, use_bias=False),
            layers.ReLU(),
            layers.Dense(nums, use_bias=False)
        ], name='input_net')
        self.state_net = tf.keras.Sequential([
            layers.LSTM(nums, return_sequences=True, use_bias=False, stateful=True,
                         batch_input_shape = (1, 1, nums), name = "lstm1",recurrent_activation='sigmoid'),
            layers.LSTM(nums, return_sequences=True, use_bias=False, stateful=True,
                        name="lstm2",recurrent_activation='sigmoid'),
        ], name='state_net')

        self.decoder_net = tf.keras.Sequential([
            layers.Dense(nums // 3, activation='relu'),
            layers.Dense(2)
        ], name='decoder_net')
    def call(self, x, training=None):
        #x(1,1,37)
        input_out = self.input_net(x)
        state_out = self.state_net(input_out)
        decoder_out = self.decoder_net(state_out)
        return [input_out, state_out, decoder_out]

class Yaw_Height_HDC_ANN:
    def __init__(self, nums, weight_path):
        self.YAW_HEIGHT_HDC_Y_DIM = nums - 1
        self.YAW_HEIGHT_HDC_H_DIM = nums - 1
        self.pi = tf.constant(np.pi , dtype = tf.float32)
        self.cann_ann_yaw = CANN1DNet(nums)
        self.cann_ann_yaw(tf.zeros((1, 1, nums)))
        self.cann_ann_yaw.load_weights(weight_path)
        self.cann_ann_yaw.reset_states()
        self.cann_ann_height= CANN1DNet(nums)
        self.cann_ann_height(tf.zeros((1, 1, nums)))
        self.cann_ann_height.load_weights(weight_path)
        self.cann_ann_height.reset_states()
        self.YAW_HEIGHT_HDC_Y_RANGE = 2 * np.pi
        self.YAW_HEIGHT_HDC_H_RANGE = self.YAW_HEIGHT_HDC_H_DIM
        self.YAW_HEIGHT_HDC_Y_RES = self.YAW_HEIGHT_HDC_Y_RANGE / self.YAW_HEIGHT_HDC_Y_DIM
        self.YAW_HEIGHT_HDC_H_RES = self.YAW_HEIGHT_HDC_H_RANGE / self.YAW_HEIGHT_HDC_H_DIM
        self.height_th = 2 * np.pi /self.YAW_HEIGHT_HDC_H_DIM
        self.theta_th = 2 * np.pi /self.YAW_HEIGHT_HDC_Y_DIM
        #通过设置累加器解决高度信息非周期性的问题
        self.height_buffer = 0
        #记录之前时刻的高度以及角度，迭代网络的同时用于触发累加器，初始时刻置为0
        self.pre_height = 0
        self.pre_yaw = 0
        self.odo_yaw = 0
        self.YAW_HEIGHT_HDC_VT_INJECT_ENERGY = 0.1
        self.cann_ann_history = []
        self.target_yaw_history = []
        self.yaw_decoded_history = []
        curYawTheta, curHeight = self.get_hdc_initial_value()
        self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = np.array([curYawTheta, curHeight])

    def get_hdc_initial_value(self):
        #高度初值和角度初值全为0
        return 0. , 0.


    def get_current_yaw_height_value(self ,cur_target_yaw_encoded, cur_target_height_encoded,heightV):
        cann_yaw_out = self.cann_ann_yaw(cur_target_yaw_encoded)
        #返回解码出的角度值,高度值
        yaw_decoder_output = tf.reshape(cann_yaw_out[-1] , (1, 2))
        yaw_decoder_output = tf.math.atan2(yaw_decoder_output[0, 0], yaw_decoder_output[0, 1])

        # self.yaw_decoded_history.append(yaw_decoder_output)
        if heightV == 0:
            height_decoder_output = self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH[1]
        else:
            cann_height_out = self.cann_ann_height(cur_target_height_encoded)
            height_decoder_output = tf.reshape(cann_height_out[-1], (1, 2))
            height_decoder_output = tf.math.atan2(height_decoder_output[0, 0], height_decoder_output[0, 1])
            height_decoder_output = tf.math.mod(height_decoder_output , 2 * self.pi) / self.height_th
        yaw_decoder_output = tf.math.mod(yaw_decoder_output , 2 * self.pi) / self.theta_th
        return yaw_decoder_output, height_decoder_output


    def yaw_height_hdc_iteration(self, vt_id, yawRotV, heightV, VT):
        #目标编码高度和目标编码角度计算，类似imu的累进形式
        cur_target_yaw = self.pre_yaw + yawRotV / self.YAW_HEIGHT_HDC_Y_RES
        self.pre_yaw = cur_target_yaw
        # self.target_yaw_history.append(cur_target_yaw)
        cur_target_height = self.pre_height + heightV / self.YAW_HEIGHT_HDC_H_RES
        self.pre_height = cur_target_height
        cur_target_height *= self.height_th
        cur_target_yaw *= self.theta_th
        cur_target_yaw_encoded = pre_encoder_keras(tf.reshape(tf.cast(cur_target_yaw ,dtype = tf.float32), (1,1)),
                                                   self.cann_ann_yaw.nums)
        cur_target_height_encoded = pre_encoder_keras(tf.reshape(tf.cast(cur_target_height ,dtype = tf.float32), (1,1)),
                                                      self.cann_ann_height.nums)
        #网络传播
        yaw_decoder_output, height_decoder_output = self.get_current_yaw_height_value(cur_target_yaw_encoded,
                                                                                      cur_target_height_encoded,
                                                                                      heightV)
        self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = np.array((yaw_decoder_output, height_decoder_output))
        if VT[vt_id].first != 1:
            act_yaw = VT[vt_id].hdc_yaw
            act_height = VT[vt_id].hdc_height
            # print(act_yaw, yaw_decoder_output)
            #根据当前值与VT值计算所需产生的偏移修正
            bais = get_delta_hdc(self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH, [act_yaw, act_height],self.YAW_HEIGHT_HDC_Y_DIM)
            #修正输出
            self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = np.mod(self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH + bais,
                                                         self.YAW_HEIGHT_HDC_Y_DIM)
            #修正内部累加器
            self.pre_yaw += bais[0]
            self.pre_height += bais[1]
        return self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH

