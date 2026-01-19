from src.cann1d_full import CANN2ANN
from src.cann_in_math import CANN1D
from matplotlib import pyplot as plt
import torch
import numpy as np
from src.func import pre_encoder,decoder_in_math,generate_cann_data

def test_noise():
    noise = np.random.normal(0 , 0.1 ,(1000 , 37))
    noise_decoded_output = decoder_in_math(noise , theta_output=False)
    x = np.arange(noise.shape[0])
    plt.figure()
    plt.plot(x , noise_decoded_output)
    plt.show()


def test_random():
    weight_path =  r"../temp/model1d_weights_37.pth"
    net = CANN2ANN(37)
    cann_net = CANN1D(37)
    net.load_state_dict(torch.load(weight_path))
    net.eval().cuda()
    data_list = generate_cann_data(cann_net, mode="test")
    with torch.no_grad():
        Iext = torch.from_numpy(np.array(data_list[0])).cuda()
        I_encoded = pre_encoder(torch.clone(Iext),net.nums).unsqueeze(1)
        ann_out_list = net(I_encoded)
        ann_input = np.array(ann_out_list[0].squeeze(1).cpu())
        cann_state = np.array(ann_out_list[1].squeeze(1).cpu())
        decoded_output = np.array(ann_out_list[2].squeeze(1).cpu())
        decoded_output = np.arctan2(decoded_output[:, 0], decoded_output[:, 1])
    x = cann_net.x
    nums = int(cann_state.shape[0] / 2)
    data_list[1] = torch.tensor(np.array(data_list[1]))
    data_list[2] = torch.tensor(np.array(data_list[2]))
    data_list[1] = torch.nn.functional.normalize(data_list[1], dim=1, eps=1e-12)
    data_list[2] = torch.nn.functional.normalize(data_list[2], dim=1, eps=1e-12)
    data_list[2] = np.array(data_list[2])
    # cann_ani = CANN1d_Animation(cann_state, data_list[2], x , decoded_output = decoded_output , ylim=(0,0.5))
    # ani = animation.FuncAnimation(cann_ani.fig, func=cann_ani.fig_update, frames=nums, init_func=cann_ani.fig_init,
    #                               interval=40, blit=False)
    # # ani.save("拟合.gif", fps=25, writer='imagemagick')
    # # plt.figure()
    # # cann_ani = CANN_Animation(data_list[1], data_list[2], x,  ylim=(0, 0.5))
    # # ani = animation.FuncAnimation(cann_ani.fig, func=cann_ani.fig_update, frames=nums, init_func=cann_ani.fig_init,
    # #                               interval=40, blit=False)
    # # # ani.save("数学形式.gif", fps=25, writer='imagemagick')
    decoded_output_math = decoder_in_math(cann_state)
    math_output = decoder_in_math(data_list[2])
    x1 = np.arange(0, decoded_output.shape[0])
    plt.figure()
    data_list[0] = np.where(data_list[0] > np.pi, data_list[0] - 2 * np.pi, data_list[0])
    plt.plot(x1, data_list[0], label="Input", color='b')
    # plt.plot(x1, decoded_output_math, label="ANN_with_math", color='r')
    plt.plot(x1, decoded_output, label="ANN", color='y')
    plt.plot(x1, math_output, label="CANN", color='g')
    plt.xlabel('Frame' , fontsize = '14')
    plt.ylabel('Angle(rad)', fontsize = '14')
    plt.title('Decoded Output' , fontsize = '14')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1) , fontsize = '10')
    plt.tight_layout()
    plt.show()
    np.savetxt("../np_txt/model_1d/Input.txt", data_list[0])
    np.savetxt("../np_txt/model_1d/ANN_decoded_output.txt", decoded_output)
    np.savetxt("../np_txt/model_1d/CANN_decoded_output.txt", math_output)