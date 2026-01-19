import matplotlib.pyplot as plt
from src.func import decoder_in_math
import brainpy.math as bm

class CANN1d_Animation:
    def __init__(self, Iext, cann_state, X, decoded_output=None, ylim = (0, 40)):
        self.I = Iext
        self.U = cann_state
        self.X = X
        if decoded_output is None:
            self.output = decoder_in_math(cann_state)
        else: self.output = decoded_output
        self.ylim = ylim
        self.fig, self.ax = plt.subplots()

    def fig_init(self , ):
        self.ax.set_ylim(self.ylim)
        line1, = self.ax.plot(self.X, self.I[0], "b" , label = "I")
        line2, = self.ax.plot(self.X, self.U[0], "g" , label = "u")
        line3 = self.ax.axvline(x = self.output[0], color='r', linestyle='-')
        self.ax.set_title('t = {0:.1f}ms'.format(0))
        return line1, line2, line3

    def fig_update(self , i):
        self.ax.cla()
        self.ax.set_ylim(self.ylim)
        ts = i * 2
        line1, = self.ax.plot(self.X , self.I[ts] , "b" , label = "I")
        line2, = self.ax.plot(self.X , self.U[ts] , "g" , label = "u")
        line3 = self.ax.axvline(x = self.output[ts], color='r', linestyle='-')
        self.ax.set_title('t = {0:.1f}ms'.format(ts * 0.1))
        self.ax.legend()
        return line1, line2, line3

class CANN3d_Animation:
    def __init__(self, Iext, U ,location,decoded_output):
        # 由于传入时按照（X,Y)、（X，Y，Z）的维度排列传入，因此绘制时需要改变维度分布方式
        self.I = bm.transpose(Iext , [0, 2, 1])
        self.U = bm.transpose(U , [0, 2, 1])
        self.location = location
        self.decoded_output = decoded_output
        self.fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        self.ax1 = axes[0 , 0]
        self.ax2 = axes[0 , 1]
        self.ax3 = self.fig.add_subplot(2, 2, 3, projection='3d')
        self.ax4 = self.fig.add_subplot(2, 2, 4, projection='3d')
        self.ax1.set_xlabel("X Position")
        self.ax1.set_ylabel("Y Position")
        self.ax2.set_xlabel("X Position")
        self.ax2.set_ylabel("Y Position")
        self.ax3.set_xlabel("X Position")
        self.ax3.set_ylabel("Y Position")
        self.ax3.set_zlabel("Z Position")
        self.ax3.set_title("Decoded Output Trajectory")
        self.ax4.set_xlabel("X Position")
        self.ax4.set_ylabel("Y Position")
        self.ax4.set_zlabel("Z Position")
        self.ax4.set_title("Trajectory")

    def fig_init(self , ):
        self.im1 = self.ax1.imshow(self.I[0], cmap='viridis', vmin=self.I.min(), vmax=self.I.max())
        self.im2 = self.ax2.imshow(self.U[0], cmap='viridis', vmin=self.U.min(), vmax=self.U.max())
        self.ax1.set_title('Iext ,t = {0:.1f}ms'.format(0))
        self.ax2.set_title('U ,t = {0:.1f}ms'.format(0))
        self.line1 = self.ax3.scatter(self.decoded_output[0, 0], self.decoded_output[0, 1],
                                      self.decoded_output[0, 2], c = "blue")
        self.line2 = self.ax4.scatter(self.location[0, 0], self.location[0, 1],
                                      self.location[0, 2], c = "red")
        self.ax3.set_xlim(self.decoded_output[:, 0].min(), self.decoded_output[:, 0].max())
        self.ax3.set_ylim(self.decoded_output[:, 1].min(), self.decoded_output[:, 1].max())
        self.ax3.set_zlim(self.decoded_output[:, 2].min(), self.decoded_output[:, 2].max())
        self.ax4.set_xlim(self.location[:, 0].min(), self.location[:, 0].max())
        self.ax4.set_ylim(self.location[:, 1].min(), self.location[:, 1].max())
        self.ax4.set_zlim(self.location[:, 2].min(), self.location[:, 2].max())
        return [self.im1 , self.im2 , self.line1, self.line2]

    def fig_update(self , i):
        self.line1.remove()
        self.line2.remove()
        ts = i * 2
        self.im1.set_array(self.I[ts])
        self.im2.set_array(self.U[ts])
        self.ax1.set_title('Iext ,t = {0:.1f}ms'.format(ts * 0.1))
        self.ax2.set_title('U ,t = {0:.1f}ms'.format(ts * 0.1))
        x = self.decoded_output[:ts, 0]
        y = self.decoded_output[:ts, 1]
        z = self.decoded_output[:ts, 2]
        self.line1 = self.ax3.scatter(x, y, z ,c="blue")
        x = self.location[:ts, 0]
        y = self.location[:ts, 1]
        z = self.location[:ts, 2]
        self.line2 = self.ax4.scatter(x, y, z, c="red")
        return [self.im1 , self.im2, self.line1, self.line2]

def plot_decoded_ouput(x, line_list, label_list, color_list):
    for idx in range(len(line_list)):
        plt.plot(x , line_list[idx], label = label_list[idx], color = color_list[idx])
    plt.legend()
    plt.show()
