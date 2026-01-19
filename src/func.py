import brainpy.math as bm
import numpy as np
import torch
import brainpy as bp
import os
import pickle
import random
def get_random_input(mode = "train",length_defined = None):
    #  case1:旋转加跳变 , case2:旋转加停止
    #训练使用短序列，降低拟合难度
    if mode == "train":
        length = bm.random.randint(3, 5)
    else:
        length = bm.random.randint(25, 30)
    dur = bm.zeros((length + 1))
    #设置多个输入时间段，单位为ms
    rand = bm.random.rand(length)
    rand = bm.where(rand < 0.5 , 0.5 , rand)
    dur[1:] = 30 * rand
    dur = bm.cumsum(dur)
    #此处对输入时间段进行细分，每0.1ms设置一个value
    duration = 10 * int( dur[-1] )
    Iext = bm.zeros( duration )
    for i in range(length - 1):
        slice_id0 = 10 * int(dur[i])
        slice_id1 = 10 * int(dur[i + 1])
        rand_num = bm.random.rand()
        if rand_num < 0.2:
            Iext[slice_id0: slice_id1] =  (2 * bm.random.rand()) * bm.pi
        elif rand_num < 0.6:
            Iext[slice_id0: slice_id1] = bm.linspace(0., 2 * bm.pi, slice_id1 - slice_id0)
        else:
            Iext[slice_id0: slice_id1] = bm.linspace(2 * bm.pi, 0 , slice_id1 - slice_id0)
    if length % 2 == 0 :
        Iext[10 * int(dur[-2]) : 10 * int(dur[-1])] = 0
    else:
        Iext[10 * int(dur[-2]): 10 * int(dur[-1])] = (2 * bm.random.rand()) * bm.pi
    Iext = Iext.reshape(-1, 1)
    if length_defined is not None:
        if Iext.shape[0] < length_defined:
            padding = (2 * bm.random.rand()) * bm.pi * bm.ones((length_defined - Iext.shape[0] , 1))
            Iext = bm.concatenate((Iext,padding), axis = 0)
        else:
            Iext = Iext[:length_defined, :]
    return Iext

def generate_cann_data(cann, mode = "train" , decoder_mode = True, Iext = None, length_defined = None):
    if Iext is None:
        Iext = get_random_input(mode,length_defined)
    Iext_encoded = cann.get_stimulus_by_pos(Iext)
    runner = bp.DSRunner(cann, inputs=['input', Iext_encoded, 'iter'], monitors=['u'], dyn_vars=cann.vars())
    runner.run(Iext.shape[0]/10)
    decoder_output = decoder_in_math(runner.mon.u, theta_output = decoder_mode)
    state = runner.mon.u
    #输入，高斯编码后的输入，状态，解码值
    return [Iext ,Iext_encoded, state, decoder_output]

def generate_3dcann_data(cann, mode="train" , decoder_mode = True ,flatten = False, Iext = None,length_defined = None):
    if Iext is None:
        Iext_x = get_random_input(mode,length_defined)
        Iext_y = get_random_input(mode,length_defined)
        Iext_z = get_random_input(mode,length_defined)
        if length_defined is None:
            Iext_length = min(Iext_x.shape[0], Iext_y.shape[0], Iext_z.shape[0])
            Iext_x = Iext_x[:Iext_length, :]
            Iext_y = Iext_y[:Iext_length, :]
            Iext_z = Iext_z[:Iext_length, :]
        Iext = bm.concatenate((Iext_x, Iext_y, Iext_z), axis=1)
    Iext_encoded = cann.get_stimulus_by_pos(Iext)
    runner = bp.DSRunner(cann, inputs=['input', Iext_encoded, 'iter'], monitors=['u'], dyn_vars=cann.vars())
    runner.run(Iext_encoded.shape[0]/ 10)
    decoded_output = cann3d_decoder_in_math(runner.mon.u, decoder_mode)
    if flatten:
        Iext_encoded = bm.flatten(Iext_encoded , start_dim=1)
        runner.mon.u = bm.flatten(runner.mon.u, start_dim=1)
    state = runner.mon.u
    return [Iext, Iext_encoded, state, decoded_output]

#(sqlength , 1) -> (sqlength , dims)
def pre_encoder( I : torch.tensor ,dims = 37)->torch.tensor:
    interval = dims - 1
    #由于首尾同位，神经元个数等于间隔+1
    I = torch.remainder(I , 2 * torch.pi)
    I = torch.where(I > torch.pi , I - 2 * torch.pi , I)
    I_encoded = (I + torch.pi) / torch.pi * interval/ 2
    I_integer = I_encoded.to(torch.int64)
    I_frac = (I_encoded - I_integer).reshape(-1 , 1)
    I_integer = I_integer.squeeze(1)
    I_encoded =  ( (1 - I_frac) * torch.nn.functional.one_hot( I_integer  , dims) +
                 I_frac * torch.nn.functional.one_hot((I_integer + 1)%dims, dims))
    I_encoded = I_encoded.to(torch.float32)
    return I_encoded

#cann_state:(sq_length , neuro_num) , output:(sq_length , 1)
def decoder_in_math(cann_state , range = (-bm.pi , bm.pi) , theta_output = True):
    #output:-pi --> pi
    if isinstance(cann_state , np.ndarray):cann_state = bm.array(cann_state)
    neuro_num = cann_state.shape[1]
    theta = bm.linspace(range[0], range[1] , neuro_num)
    #直接给出结果，或者三角编码后的值
    if theta_output:
        #考虑到首尾两个神经元的方向实际上是一个神经元，因此去除尾部神经元
        output = bm.arctan2(bm.sum(cann_state[:, :-1] * bm.sin(theta[:-1]), axis = 1) ,
                            bm.sum(cann_state[:, :-1] * bm.cos(theta)[:-1], axis = 1))
        output = output.reshape(-1, 1)
    else:
        output = bm.stack((bm.sum(cann_state[:, :-1] * bm.sin(theta[:-1]), axis = 1),
                           bm.sum(cann_state[:, :-1] * bm.cos(theta[:-1]), axis = 1)) , axis=1)
        output = output.reshape(-1, 2)
    # #改变解码范围至（o , 2 *pi)
    # output = bm.mod(output , 2 * bm.pi)
    return output

def cann3d_decoder_in_math(U , theta_output = True):
    U_x = bm.sum(bm.sum(U , axis =-1), axis=-1)
    U_y = bm.sum(bm.sum(U , axis =-1), axis=1)
    U_z = bm.sum(bm.sum(U , axis = 1), axis=1)
    out_x = decoder_in_math(U_x, theta_output = theta_output)
    out_y = decoder_in_math(U_y, theta_output= theta_output)
    out_z = decoder_in_math(U_z, theta_output= theta_output)
    out = bm.concatenate((out_x , out_y , out_z) , axis=1)
    return out

def cann3d_pre_encoder( I : torch.tensor , dims = 37, concat = False, range = ())->torch.tensor:
    I_x = I[: , 0].reshape(-1 , 1)
    I_y = I[: , 1].reshape(-1 , 1)
    I_z = I[: , 2].reshape(-1 , 1)
    if concat:
        I_x_decoded = pre_encoder(I_x ,dims)
        I_y_decoded = pre_encoder(I_y, dims)
        I_z_decoded = pre_encoder(I_z, dims)
        I_encoded = torch.concatenate((I_x_decoded,I_y_decoded,I_z_decoded), dim = 1)
        return I_encoded
    I_x_decoded = pre_encoder(I_x ,dims).reshape(-1, dims, 1, 1)
    I_y_decoded = pre_encoder(I_y, dims).reshape(-1, 1, dims, 1)
    I_z_decoded = pre_encoder(I_z, dims).reshape(-1, 1, 1, dims)
    I_encoded = I_x_decoded*I_y_decoded*I_z_decoded
    return I_encoded

def cann1d_data_save(cann , length_defined , save_dir , filename):
    data_list = generate_cann_data(cann, "train", decoder_mode=False,length_defined = length_defined)
    data_dict = {"I":data_list[0], "input_weight":data_list[1],
                 "state":data_list[2], "output":data_list[3]}
    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump(data_dict, f)
def cann3d_data_save(cann, length_defined, save_dir, filename, flatten = True):
    data_list = generate_3dcann_data(cann, "train", decoder_mode=False,flatten = flatten,length_defined = length_defined)
    if not flatten:
        #input_weight,state:dimension(sq,x,y,z)
        input_x = bm.sum(bm.sum(data_list[1], axis=-1), axis=-1)
        input_y = bm.sum(bm.sum(data_list[1], axis=-1), axis=1)
        input_z = bm.sum(bm.sum(data_list[1], axis=1), axis=1)
        data_list[1] = bm.concatenate((input_x,input_y,input_z), axis = 1)
        state_x = bm.sum(bm.sum(data_list[2], axis=-1), axis=-1)
        state_y = bm.sum(bm.sum(data_list[2], axis=-1), axis=1)
        state_z = bm.sum(bm.sum(data_list[2], axis=1), axis=1)
        data_list[2] = bm.concatenate((state_x,state_y,state_z), axis = 1)
    data_dict = {"I":data_list[0], "input_weight":data_list[1],
                 "state":data_list[2], "output":data_list[3]}
    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump(data_dict, f)

def cann3d_data_save_mixed(cann1, cann2, save_dir, filename):
    data_list1 = generate_3dcann_data(cann1, "train", decoder_mode=False, flatten=True)
    data_list2 = generate_3dcann_data(cann2, decoder_mode = False, flatten=True, Iext = data_list1[0])
    data_dict = {"I":data_list1[0], "input_weight":data_list1[1],
                 "state":data_list2[2], "output":data_list2[3]}
    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump(data_dict, f)

def get_data(filepath , mode = "dict"):
    with open(filepath, 'rb') as f:  # 二进制读取模式 'rb'
        loaded_data = pickle.load(f)
    for k in loaded_data:
        loaded_data[k] = np.array(loaded_data[k])
    if mode == "list":
        data_list = []
        for k in loaded_data:
            data_list.append(loaded_data[k])
        return data_list
    return loaded_data

if __name__ == "__main__":
    from cann3d_in_math import CANN3D
    from cann_in_math import CANN1D
    cann = CANN1D(37)
    res_list = generate_cann_data(cann)
    res_list[0] = bm.where(res_list[0] > bm.pi , res_list[0] - 2*bm.pi , res_list[0])
    res_list[0] = (res_list[0] + bm.pi) * 18 / bm.pi
    res_list[-1] = (res_list[-1] + bm.pi) * 18 / bm.pi
    res_out = bm.concatenate((res_list[0] , res_list[-1]), axis=1)
    print(res_out)
