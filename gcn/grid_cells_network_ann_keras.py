import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras import Model,layers,Sequential
from my_func.func_keras import cann3d_pre_encoder_keras ,get_delta_gcn
class CANN3DNet(Model):
    def __init__(self,nums):
        super(CANN3DNet, self).__init__()
        self.nums = nums
        self.tri_nums = nums * 3
        self.input_net = Sequential([
            layers.Dense(self.tri_nums, use_bias=False, name='input_net_0'),
            layers.ReLU(name='input_net_1'),
            layers.Dense(self.tri_nums, use_bias=False, name='input_net_2')
        ], name='input_net')

        self.state_net = Sequential([
            layers.LSTM(self.tri_nums, stateful=True, return_sequences=True, use_bias=False,
                        batch_input_shape=(1, 1, self.tri_nums), name='lstm_1',recurrent_activation='sigmoid'),
            layers.LSTM(self.tri_nums, stateful=True, return_sequences=True, use_bias=False,
                        batch_input_shape=(1, 1, self.tri_nums), name='lstm_2',recurrent_activation='sigmoid'),
        ], name='lstm_net')

        self.decoder_net = tf.keras.Sequential([
            layers.Dense(self.nums, activation='relu'),
            layers.Dense(6)
        ], name='decoder_net')

    def call(self, x, training=None):
       input_out = self.input_net(x)
       state_out = self.state_net(input_out)
       decoder_out = self.decoder_net(state_out)

       return [input_out, state_out, decoder_out]
class GridCellNetwork:
    def __init__(self, nums ,weight_path):
        self.pi = tf.constant(np.pi , dtype = tf.float32)
        self.GC_X_DIM = nums - 1
        self.GC_Y_DIM = nums - 1
        self.GC_Z_DIM = nums - 1
        #19 nueros only have 18 intervals for the identity between the first one and the last one
        self.GC_X_TH_SIZE = 2 * np.pi / self.GC_X_DIM
        self.GC_Y_TH_SIZE = 2 * np.pi / self.GC_Y_DIM
        self.GC_Z_TH_SIZE = 2 * np.pi / self.GC_Z_DIM

        self.GC_TH_SIZE = tf.constant([self.GC_X_TH_SIZE, self.GC_Y_TH_SIZE, self.GC_Z_TH_SIZE])
        self.GC_TH_SIZE = tf.reshape(self.GC_TH_SIZE, (1,3))
        #cann2ann
        self.cann_ann = CANN3DNet(nums)
        self.cann_ann(tf.zeros((1, 1, nums * 3)))
        self.cann_ann.load_weights(weight_path)
        self.cann_ann.reset_states()
        self.buffer = [0, 0, 0]
        gcX, gcY, gcZ = self.get_gc_initial_pos()
        self.MAX_ACTIVE_XYZ_PATH =[gcX, gcY, gcZ]
        self.pre_location = np.array(self.MAX_ACTIVE_XYZ_PATH)

    def get_gc_initial_pos(self):
        gcX = int(np.floor((self.GC_X_DIM - 1)/ 2))  # in 1:GC_X_DIM
        gcY = int(np.floor((self.GC_Y_DIM - 1) / 2))  # in 1:GC_Y_DIM
        gcZ = int(np.floor((self.GC_Z_DIM - 1)/ 2))  # in 1:GC_Z_DIM
        return gcX, gcY, gcZ


    def get_gc_xyz(self,cur_location_encoded ):
        cur_location_encoded = cann3d_pre_encoder_keras(cur_location_encoded, self.cann_ann.nums, concat=True)
        cann_out = self.cann_ann(cur_location_encoded)
        pre_decoded_out = tf.reshape(cann_out[-1], (1, 6))
        #get the output in radian
        gc_x = tf.math.atan2(pre_decoded_out[0, 0], pre_decoded_out[0, 1])
        gc_y = tf.math.atan2(pre_decoded_out[0, 2], pre_decoded_out[0, 3])
        gc_z = tf.math.atan2(pre_decoded_out[0, 4], pre_decoded_out[0, 5])
        #from radian to idx
        gc_x = tf.math.mod(gc_x, 2 * self.pi) / self.GC_X_TH_SIZE
        gc_y = tf.math.mod(gc_y, 2 * self.pi) / self.GC_Y_TH_SIZE
        gc_z = tf.math.mod(gc_z, 2 * self.pi) / self.GC_Z_TH_SIZE
        return gc_x,gc_y,gc_z


    def gc_iteration(self, vt_id, transV, curYawThetaInRadian, heightV, VT):
        """
        updatae_steps:
            1.get the location which needs to be encoded
            2.ANN forward
            3.decoding state
        """
        transV = tf.cast(transV , dtype = tf.float32)
        delta_x = transV * np.cos(curYawThetaInRadian)
        delta_y = transV * np.sin(curYawThetaInRadian)
        delta_z = heightV
        delta = np.array([delta_x, delta_y, delta_z])
        cur_location = self.pre_location + delta
        self.pre_location = cur_location
        cur_location_encoded = cur_location * self.GC_TH_SIZE
        gc_x,gc_y,gc_z = self.get_gc_xyz(cur_location_encoded)
        self.MAX_ACTIVE_XYZ_PATH = np.array([gc_x, gc_y, gc_z])
        if VT[vt_id].first != 1:
            #获得VT记录的索引
            actX = np.mod(VT[vt_id].gc_x, self.GC_X_DIM)
            actY = np.mod(VT[vt_id].gc_y, self.GC_Y_DIM)
            actZ = np.mod(VT[vt_id].gc_z, self.GC_Z_DIM)
            bais = get_delta_gcn(self.MAX_ACTIVE_XYZ_PATH, [actX, actY, actZ], self.GC_X_DIM)
            bais = tf.cast(bais , dtype = tf.float32)
            self.MAX_ACTIVE_XYZ_PATH = np.mod(self.MAX_ACTIVE_XYZ_PATH + bais, self.GC_X_DIM)
            self.pre_location += bais

        return self.MAX_ACTIVE_XYZ_PATH.tolist()
