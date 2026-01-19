import numpy as np
import torch

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
def decoder_in_math(cann_state , range = (-np.pi , np.pi) , theta_output = True):
    #output:-pi --> pi
    neuro_num = cann_state.shape[1]
    theta = np.linspace(range[0], range[1] , neuro_num)
    #直接给出结果，或者三角编码后的值
    if theta_output:
        #考虑到吴思模型首尾两个神经元的方向实际上是一个神经元，因此去除尾部神经元
        output = np.arctan2(np.sum(cann_state[:, :-1] * np.sin(theta[:-1]), axis = 1) ,
                            np.sum(cann_state[:, :-1] * np.cos(theta)[:-1], axis = 1))
        output = output.reshape(-1, 1)
    else:
        output = np.stack((np.sum(cann_state[:, :-1] * np.sin(theta[:-1]), axis = 1),
                           np.sum(cann_state[:, :-1] * np.cos(theta[:-1]), axis = 1)) , axis=1)
        output = output.reshape(-1, 2)
    return output

def cann3d_decoder_in_math(U , theta_output = True):
    U_x = np.sum(np.sum(U , axis =-1), axis=-1)
    U_y = np.sum(np.sum(U , axis =-1), axis=1)
    U_z = np.sum(np.sum(U , axis = 1), axis=1)
    out_x = decoder_in_math(U_x, theta_output = theta_output)
    out_y = decoder_in_math(U_y, theta_output= theta_output)
    out_z = decoder_in_math(U_z, theta_output= theta_output)
    out = np.concatenate((out_x , out_y , out_z) , axis=1)
    return out

def cann3d_pre_encoder( I : torch.tensor , dims = 37, concat = False)->torch.tensor:
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

def vt_modle_hdc(x):
    m = 0.2 * np.exp(-x ** 2 / 9.1)
    return (m / (1 + m) * x)


def vt_modle_gcn(x):
    m = 0.13 * np.exp(-x ** 2 / 10.98)
    return (m / (1 + m) * x)
def find_short_delta(start, end):
    delta = end - start
    delta = delta if abs(delta) < 36 - abs(delta) else np.sign(delta) * (abs(delta) - 36)
    return delta
def find_short_vector(start, end, dims = 36):
    delta = []
    for i,j in zip(start, end):
        delta.append(find_short_delta(i , j))
    # 找到解码值指向实际值的最短向量
    return np.array(delta)


def get_delta_hdc(start, end, dims = 36):
    vec = find_short_vector(start, end, dims)
    vector_norm = np.linalg.norm(vec)
    bais =np.array((0, 0)) if vector_norm > 9 else vt_modle_hdc(vector_norm) * vec / vector_norm
    return bais

def get_delta_gcn(start, end, dims = 36):
    vec = find_short_vector(start, end, dims)
    vector_norm = np.linalg.norm(vec)
    bais = np.array((0, 0, 0)) if vector_norm > 9 else vt_modle_hdc(vector_norm) * vec / vector_norm

    return bais

if __name__ == "__main__":
    pass