import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras import Model,layers,Sequential
from gcn.grid_cells_network_ann import CANN3Dnet
from hdcn.yaw_height_hdc_ann import CANNnet

class CANN1DNet(Model):
    def __init__(self,nums):
        super(CANN1DNet, self).__init__()
        self.nums = nums
        self.input_net = tf.keras.Sequential([
            layers.Dense(self.nums, use_bias=False),
            layers.ReLU(),
            layers.Dense(self.nums, use_bias=False)
        ], name='input_net')

        self.state_net = tf.keras.Sequential([
            layers.LSTM(self.nums, return_sequences=True, use_bias=False, stateful=True,
                         batch_input_shape = (1, 1, self.nums), name = "lstm1",recurrent_activation='sigmoid'),
            layers.LSTM(self.nums, return_sequences=True, use_bias=False, stateful=True,
                        name="lstm2",recurrent_activation='sigmoid'),
        ], name='state_net')

        self.decoder_net = tf.keras.Sequential([
            layers.Dense(self.nums // 3, activation='relu'),
            layers.Dense(2)
        ], name='decoder_net')

    def call(self, x):
        #x(1,1,37)
        print(x[0 , 0,:5])
        input_out = self.input_net(x)
        state_out = self.state_net(input_out)
        decoder_out = self.decoder_net(state_out)
        return [input_out, state_out, decoder_out]
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

    def call(self, x):
        print(x[0,0,:5])
        input_out = self.input_net(x)
        state_out = self.state_net(input_out)
        decoder_out = self.decoder_net(state_out)
        return [input_out, state_out, decoder_out]



class CANNWeightManager:
    def __init__(self, nums, mode='1D'):
        """
        mode: '1D' 对应 CANN1DNet, '3D' 对应 CANN3DNet
        """
        self.nums = nums
        self.mode = mode
        # 根据模式确定 LSTM 内部的维度
        self.target_nums = nums if mode == '1D' else nums * 3

    def _reorder_lstm_gate(self, w):
        """将 PyTorch 的 IFGO 顺序转换为 Keras 的 ICFO 顺序"""
        n = self.target_nums
        # w 传入时已经是 (input_dim, 4*n)
        i = w[:, :n]
        f = w[:, n:2 * n]
        g = w[:, 2 * n:3 * n]
        o = w[:, 3 * n:]
        # Keras 顺序: i, c(即g), f, o
        return np.concatenate([i, g, f, o], axis=1)

    def convert_and_save(self, pth_path, keras_model, h5_path,torch_model):
        """解析 pth，注入 keras_model，并保存为 h5"""
        print(f"[*] 正在从 {pth_path} 提取权重 (模式: {self.mode})...")
        pt_state = torch.load(pth_path, map_location='cpu')

        # 1. Input Net 转换
        keras_model.input_net.layers[0].set_weights([pt_state['input_net.0.weight'].numpy().T])
        keras_model.input_net.layers[2].set_weights([pt_state['input_net.2.weight'].numpy().T])

        # 2. State Net (LSTM) 转换
        # 注意：CANN1DNet 里的 Sequential 叫 state_net, CANN3DNet 里的叫 lstm_net (根据你提供的代码)
        state_layer_name = 'state_net'
        state_container = getattr(keras_model, state_layer_name)

        for l in range(2):
            w_ih = pt_state[f'state_net.weight_ih_l{l}'].numpy().T
            w_hh = pt_state[f'state_net.weight_hh_l{l}'].numpy().T

            w_ih_fixed = self._reorder_lstm_gate(w_ih)
            w_hh_fixed = self._reorder_lstm_gate(w_hh)
            state_container.layers[l].set_weights([w_ih, w_hh])
            # state_container.layers[l].set_weights([w_ih, w_hh])

        # 3. Decoder Net 转换
        keras_model.decoder_net.layers[0].set_weights([
            pt_state['decoder_net.0.weight'].numpy().T,
            pt_state['decoder_net.0.bias'].numpy()
        ])
        keras_model.decoder_net.layers[1].set_weights([
            pt_state['decoder_net.2.weight'].numpy().T,
            pt_state['decoder_net.2.bias'].numpy()
        ])

        # 4. 保存为 H5
        keras_model.save_weights(h5_path)
        print(f"[+] 权重已保存至: {h5_path}")
        test_val = np.random.rand(1, 1, self.target_nums).astype(np.float32)
        res1_t = np.dot(test_val, pt_state['input_net.0.weight'].numpy().T)
        res1_t[res1_t < 0] = 0  # 模拟 ReLU
        res2_t = np.dot(res1_t, pt_state['input_net.2.weight'].numpy().T)

        # Keras 模拟 (手动提取权重)
        w0_k = keras_model.input_net.layers[0].get_weights()[0]
        w2_k = keras_model.input_net.layers[2].get_weights()[0]
        res1_k = np.dot(test_val, w0_k)
        res1_k[res1_k < 0] = 0
        res2_k = np.dot(res1_k, w2_k)

        print(f"Manual 2-layer Match: {np.allclose(res2_t, res2_k)}")
        print(f"\n[*] 开始一致性测试 ({self.mode})...")
        actual_w0 = keras_model.input_net.layers[0].weights[0].numpy()
        expected_w0 = pt_state['input_net.0.weight'].numpy().T

        print(f"权重一致性检查: {np.allclose(actual_w0, expected_w0)}")
        input_dim = self.nums if self.mode == '1D' else self.nums * 3
        test_input = np.random.rand(1, 1, input_dim).astype(np.float32)
        print(test_input[0 ,0 ,:5])
        keras_model.state_net.reset_states()
        # 1. 强制 Keras 模型构建并获取输出 Tensor
        # 在 TF 1.14 中，我们通过模型直接调用获取符号张量
        k_input_tensor = tf.cast(test_input , tf.float32)
        k_output_tensors = keras_model(k_input_tensor)
        k_vals = list(np.array(x) for x in k_output_tensors)

        # 3. Torch 推理（保持不变）
        torch_model.eval()
        with torch.no_grad():
            t_in = torch.from_numpy(test_input)
            x_in = torch_model.input_net(t_in)
            x_state, _ = torch_model.state_net(x_in)
            x_out = torch_model.decoder_net(x_state)
            t_results = [x_in.numpy(), x_state.numpy(), x_out.numpy()]
        # 4. 对比
        layers_names = ["Input_Net", "State_Net", "Decoder_Net"]
        for i in range(3):
            k_val = k_vals[i].flatten()
            t_val = t_results[i].flatten()
            diff = np.max(np.abs(t_val - k_val))
            print(f"Torch Input_Net 前5位: {t_results[1].flatten()[:5]}")
            print(f"Keras Input_Net 前5位: {k_vals[1].flatten()[:5]}")
            print(f"-> {layers_names[i]} 最大绝对误差: {diff:.2e}")
        return keras_model


if __name__ == "__main__":
    tf.enable_eager_execution()
    nums = 25
    t_model_3d = CANN3Dnet(nums)  # 假设这是你的 Torch 类
    weight_path = 'model3d_weights_25.pth'
    t_model_3d.load_state_dict(torch.load(weight_path))
    k_model_3d = CANN3DNet(nums)
    dummy_in = np.zeros((1, 1, 3*nums), dtype=np.float32)

    # 使用模型对象直接调用，而不是 predict，这在 TF 1.x 中更稳定
    _ = k_model_3d(tf.convert_to_tensor(dummy_in))

    # --- 3. 转换与保存 ---
    manager = CANNWeightManager(nums, mode='3D')
    k_model_1d = manager.convert_and_save(weight_path, k_model_3d, 'model3d_weights_25_keras.h5',t_model_3d)

