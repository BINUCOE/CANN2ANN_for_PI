import torch
import glob
from src.func import pre_encoder,get_data
from torch.optim import lr_scheduler
import os
import random
from torch.utils.data import Dataset, DataLoader


class CANN2ANN(torch.nn.Module):
    def __init__(self,nums):
        super(CANN2ANN , self).__init__()
        self.nums = nums
        self.input_net = torch.nn.Sequential(torch.nn.Linear(nums, nums,bias = False),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(nums, nums,bias = False),
                                            )
        self.state_net = torch.nn.LSTM(input_size=nums, hidden_size=nums, num_layers = 2 , bias=False)

        self.decoder_net = torch.nn.Sequential(torch.nn.Linear(nums, nums // 3),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(nums // 3, 2),
                                           )
    def forward(self , x):
        input = self.input_net(x)
        state,(hn,cn) = self.state_net(input)
        decoder_output = self.decoder_net(state)
        #( batch_size , dims)
        return [input,state,decoder_output]

    def compute_loss(self , net_out, cann_out, out_id ):
        loss = torch.nn.MSELoss(reduction="mean")(net_out[out_id], cann_out)
        return loss
class CANNDataset(Dataset):
    def __init__(self, file_list,out_id):
        self.file_list = file_list
        self.inner_id = out_id

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data_list = get_data(self.file_list[idx], mode="list")
        cur_data = data_list[self.inner_id]
        cur_data = torch.tensor((cur_data))/10
        I = torch.from_numpy(data_list[0])
        return I, cur_data

def init_weights(m):
    if  isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

def train(net, num_epochs ,out_id, save_dir ,datadir, trainer_list, scheduler_list,weights_name,batch_size = 10,val_split = 0.2):
    net.cuda()
    search = os.path.join(datadir, '*.pkl')
    listing = glob.glob(search)
    random.shuffle(listing)
    split_idx = int(len(listing) * (1 - val_split))
    train_files = listing[:split_idx]
    val_files = listing[split_idx:]
    train_loader = DataLoader(CANNDataset(train_files, out_id + 1), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CANNDataset(val_files, out_id + 1), batch_size=batch_size, shuffle=False)
    for epoch in range(num_epochs):
        net.train()
        train_loss_sum = 0
        for I, cann_out_batch in train_loader:
            for trainer in trainer_list:
                trainer.zero_grad()
            # 获取数学形式下的状态和解码结果
            I = I.cuda()
            cann_out_batch = cann_out_batch.cuda()
            I = I.reshape(I.shape[0] * I.shape[1], I.shape[2])
            # 将模拟输入预编码后输入到神经网络中
            I_encoded = pre_encoder(I, net.nums)
            I_encoded = I_encoded.reshape(cann_out_batch.shape[0], cann_out_batch.shape[1], I_encoded.shape[1])
            ann_out = net(I_encoded)
            train_loss = net.compute_loss(ann_out, cann_out_batch, out_id)
            train_loss.backward()
            for trainer in trainer_list:
                trainer.step()
            train_loss_sum += train_loss.item()
        for scheduler in scheduler_list:
            scheduler.step()
        net.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for I_val, cann_out_val_batch in val_loader:
                I_val = I_val.cuda()
                cann_out_val_batch = cann_out_val_batch.cuda()
                I_val = I_val.reshape(I_val.shape[0] * I_val.shape[1], I_val.shape[2])
                # 将模拟输入预编码后输入到神经网络中
                I_val_encoded = pre_encoder(I_val, net.nums)
                I_val_encoded = I_val_encoded.reshape(cann_out_val_batch.shape[0], cann_out_val_batch.shape[1],
                                                      I_val_encoded.shape[1])
                ann_val_out = net(I_val_encoded)
                val_loss = net.compute_loss(ann_val_out, cann_out_val_batch, out_id)
                val_loss_sum += val_loss.item()
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_val_loss = val_loss_sum / len(val_loader)
        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        if epoch % 10 == 9:
            print('weights saved')
            torch.save(net.state_dict(), os.path.join(save_dir, weights_name))
    torch.save(net.state_dict(), os.path.join(save_dir, weights_name))



if __name__ == "__main__":
    save_dir = "../temp"
    datadir = "../dataset_1d_49"
    weights_name = "model1d_weights_49.pth"
    mode = 1
    out_id = 2
    num_epochs , lr = 200, 0.001
    net = CANN2ANN(49)
    if mode == 0:
        net.apply(init_weights)
    else:
        net.load_state_dict(torch.load(os.path.join(save_dir, weights_name)))
    net_layer_list = [net.input_net , net.state_net , net.decoder_net]
    for i in range(len(net_layer_list)):
        if i == out_id:
            continue
        for param in net_layer_list[i].parameters():
            param.requires_grad = False
    trainer1 = torch.optim.Adam(list(net_layer_list[out_id].parameters()), lr=lr)
    scheduler1 = lr_scheduler.OneCycleLR(trainer1, max_lr=lr, total_steps=num_epochs)
    train(net, num_epochs, out_id, save_dir, datadir, [trainer1], [scheduler1],weights_name)