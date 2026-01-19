import pickle
from src.func import decoder_in_math
from matplotlib import animation
from src.animation import CANN1d_Animation,plot_decoded_ouput
import numpy as np
import matplotlib.pyplot as plt

def test_verify_data():
    num_nueros = 49
    with open('../dataset_1d_49/cann1d_dict0452.pkl', 'rb') as f:  # 二进制读取模式 'rb'
        loaded_dict = pickle.load(f)
    for k in loaded_dict:
        loaded_dict[k] = np.array(loaded_dict[k])
    I = loaded_dict["I"]
    I = np.where(I > np.pi, I - 2 * np.pi, I)
    Iext_x = loaded_dict["input_weight"][:, :num_nueros]
    decoded_input = decoder_in_math(Iext_x)
    state = loaded_dict["state"][:, :num_nueros]
    nums = int(state.shape[0] / 2)
    plt_x = np.arange(Iext_x.shape[0])
    ani_x = np.linspace(-np.pi, np.pi, num_nueros)
    I_x = I[:, 0]
    decoded_output_x = np.arctan2(loaded_dict["output"][:, 0], loaded_dict["output"][:, 1])
    line_list = [I_x, decoded_output_x, decoded_input]
    # 验证解码输出和输入编码再解码与输入之间的差异性，在细胞数较少分辨率不足时容易显现偏差
    label = ["I_x", "decoded_output_x", "decoded_input"]
    color = ['r', 'g', 'y']
    print(np.min(I_x - decoded_output_x))
    plot_decoded_ouput(plt_x, line_list, label, color)
    cann_ani = CANN1d_Animation(Iext_x, state, ani_x, decoded_output_x, ylim=(0, 20))
    ani = animation.FuncAnimation(cann_ani.fig, func=cann_ani.fig_update, frames=nums, init_func=cann_ani.fig_init,
                                  interval=40, blit=False)
    plt.show()