from hdcn.yaw_height_hdc_network import YawHeightHDCNetwork
from gcn.grid_cells_network import GridCellNetwork
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
class CANN2d(YawHeightHDCNetwork):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.pre_state = 0

    def hdc_iteration(self, act_height, act_yaw, energy = None, lock = False):
        """
        Pose cell update steps
        1. Add view template energy
        2. Local excitation
        3. Local inhibition
        4. Global inhibition
        5. Normalisation
        6. Path Integration (yawRotV then heightV)
        """
        # If this isn't a new visual template then add the energy at its associated pose cell location
        if energy is not None:
            self.YAW_HEIGHT_HDC[act_yaw, act_height] += energy
        # Local excitation: yaw_height_hdc_local_excitation = yaw_height_hdc elements * yaw_height_hdc weights
        yaw_height_hdc_local_excit_new = np.zeros((self.YAW_HEIGHT_HDC_Y_DIM, self.YAW_HEIGHT_HDC_H_DIM))
        for h in range(self.YAW_HEIGHT_HDC_H_DIM):
            for y in range(self.YAW_HEIGHT_HDC_Y_DIM):
                if self.YAW_HEIGHT_HDC[y, h] != 0:
                    yaw_height_hdc_local_excit_new[
                        np.ix_(self.YAW_HEIGHT_HDC_EXCIT_Y_WRAP[y: y + self.YAW_HEIGHT_HDC_EXCIT_Y_DIM],
                               self.YAW_HEIGHT_HDC_EXCIT_H_WRAP[h: h + self.YAW_HEIGHT_HDC_EXCIT_H_DIM])] += \
                        self.YAW_HEIGHT_HDC[y, h] * self.YAW_HEIGHT_HDC_EXCIT_WEIGHT
        self.YAW_HEIGHT_HDC = yaw_height_hdc_local_excit_new

        # Local inhibition: yaw_height_hdc_local_inhibition = hdc - hdc elements * hdc_inhib weights
        yaw_height_hdc_local_inhib_new = np.zeros((self.YAW_HEIGHT_HDC_Y_DIM, self.YAW_HEIGHT_HDC_H_DIM))
        for h in range(self.YAW_HEIGHT_HDC_H_DIM):
            for y in range(self.YAW_HEIGHT_HDC_Y_DIM):
                if self.YAW_HEIGHT_HDC[y, h] != 0:
                    yaw_height_hdc_local_inhib_new[
                        np.ix_(self.YAW_HEIGHT_HDC_INHIB_Y_WRAP[y: y + self.YAW_HEIGHT_HDC_INHIB_Y_DIM],
                               self.YAW_HEIGHT_HDC_INHIB_H_WRAP[h: h + self.YAW_HEIGHT_HDC_INHIB_H_DIM])] += \
                        self.YAW_HEIGHT_HDC[y, h] * self.YAW_HEIGHT_HDC_INHIB_WEIGHT

        self.YAW_HEIGHT_HDC -= yaw_height_hdc_local_inhib_new

        # Global inhibition   PC_gi = PC_li elements - inhibition
        self.YAW_HEIGHT_HDC = np.where(self.YAW_HEIGHT_HDC >= self.YAW_HEIGHT_HDC_GLOBAL_INHIB,
                                       self.YAW_HEIGHT_HDC - self.YAW_HEIGHT_HDC_GLOBAL_INHIB, 0)

        # Normalisation
        total = np.sum(self.YAW_HEIGHT_HDC)
        self.YAW_HEIGHT_HDC /= total

        self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = self.get_current_yaw_height_value()
        # self.yaw_state.append(np.sum(self.YAW_HEIGHT_HDC , axis = 1))
        if lock:
            self.YAW_HEIGHT_HDC = self.pre_state.copy()
        else:
            self.pre_state = self.YAW_HEIGHT_HDC.copy()
        return self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH

    def get_hdc_initial_value(self):
        curYawTheta = 18
        curHeight = 18
        return curYawTheta, curHeight

class CANN3d(GridCellNetwork):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.pre_state = 0

    def cann3d_iteration(self, i, j ,k, energy = None, lock = False):
        if energy is not None:
            self.GRIDCELLS[i, j, k] +=0.0939
        # Local excitation      GC_local_excitation = GC elements * GC weights
        gridcell_local_excit_new = np.zeros((self.GC_X_DIM, self.GC_Y_DIM, self.GC_Z_DIM))
        for z in range(self.GC_Z_DIM):
            for x in range(self.GC_X_DIM):
                for y in range(self.GC_Y_DIM):
                    if self.GRIDCELLS[x, y, z] != 0:
                        gridcell_local_excit_new[np.ix_(self.GC_EXCIT_X_WRAP[x: x + self.GC_EXCIT_X_DIM],
                                                        self.GC_EXCIT_Y_WRAP[y: y + self.GC_EXCIT_Y_DIM],
                                                        self.GC_EXCIT_Z_WRAP[z: z + self.GC_EXCIT_Z_DIM])] += \
                            self.GRIDCELLS[x, y, z] * self.GC_EXCIT_WEIGHT

        self.GRIDCELLS = gridcell_local_excit_new

        # Local inhibition      GC_li = GC_le - GC_le elements * GC weights
        gridcell_local_inhib_new = np.zeros((self.GC_X_DIM, self.GC_Y_DIM, self.GC_Z_DIM))
        for z in range(self.GC_Z_DIM):
            for x in range(self.GC_X_DIM):
                for y in range(self.GC_Y_DIM):
                    if self.GRIDCELLS[x, y, z] != 0:
                        gridcell_local_inhib_new[np.ix_(self.GC_INHIB_X_WRAP[x: x + self.GC_INHIB_X_DIM],
                                                        self.GC_INHIB_Y_WRAP[y: y + self.GC_INHIB_Y_DIM],
                                                        self.GC_INHIB_Z_WRAP[z: z + self.GC_INHIB_Z_DIM])] += \
                            self.GRIDCELLS[x, y, z] * self.GC_INHIB_WEIGHT
        self.GRIDCELLS -= gridcell_local_inhib_new
        # Global inhibition     Gc_gi = GC_li elements - inhibition
        self.GRIDCELLS = np.where(self.GRIDCELLS >= self.GC_GLOBAL_INHIB, self.GRIDCELLS - self.GC_GLOBAL_INHIB, 0)
        # Normalisation
        total = np.sum(self.GRIDCELLS)
        self.GRIDCELLS /= total
        self.MAX_ACTIVE_XYZ_PATH = self.get_gc_xyz()
        if lock:
            self.GRIDCELLS = self.pre_state.copy()
        else:
            self.pre_state = self.GRIDCELLS.copy()
        return self.MAX_ACTIVE_XYZ_PATH
def model_func(x, a , b):
    m = a * np.exp(-x**2 / b)
    return (m/(1 + m) * x)

def generate_data_3d(save_path):
    CANNnet1 = CANN3d()
    CANNnet2 = CANN3d()
    output_list = []
    for i in range(10):
        output1 = CANNnet1.cann3d_iteration(0, 0, 0)
        output2 = CANNnet2.cann3d_iteration(0, 0, 0)
    for i in range(8 , 26):
        print(i)
        for j in range(8, 26):
            for k in range(8, 26):
                output1 = CANNnet1.cann3d_iteration(i, j, k, energy = 0.0939,lock = True)
                output_list.append(output1)

    output_list = np.array(output_list)
    np.savetxt(save_path, output_list)

def generate_data_2d(save_path):
    CANNnet1 = CANN2d(YAW_HEIGHT_HDC_EXCIT_Y_DIM = 9, YAW_HEIGHT_HDC_EXCIT_H_DIM = 9)
    CANNnet2 = CANN2d(YAW_HEIGHT_HDC_EXCIT_Y_DIM = 9, YAW_HEIGHT_HDC_EXCIT_H_DIM = 9)
    output_list = []
    for i in range(10):
        output1 = CANNnet1.hdc_iteration(0,0)
        output2 = CANNnet2.hdc_iteration(0, 0)
    for j in range(36):
        for k in range(36):
            output1 = CANNnet1.hdc_iteration(j, k, energy = 0.0939,lock = True)
            output2 = CANNnet2.hdc_iteration(0, 0,lock = True)
            output_list.append(output1 + output2)
    output_list = np.array(output_list)
    np.savetxt(save_path, output_list)

def fit_2d(file_path):
    output_list = np.loadtxt(file_path)
    output_list1 = output_list[:, :2] - 18
    x = np.zeros((36, 36))
    for i in range(36):
        for j in range(36):
            x[i, j] = np.linalg.norm(np.array((i - 18, j - 18)))
    x_data = x.reshape(36 * 36)
    output_list_delta = np.linalg.norm(output_list1, axis=1).reshape(36, 36)
    y_data = output_list_delta.reshape(36 * 36)
    idx = y_data > 0.01
    y_data = y_data[idx]
    initial_guess = [0.3, 8.4]
    unique_x, inverse_indices = np.unique(x_data[idx], return_inverse=True)
    mean_y = np.zeros(len(unique_x))
    for i in range(len(unique_x)):
        mask = (inverse_indices == i)
        mean_y[i] = y_data[mask].mean()
    # 拟合参数
    params, covariance, infodict, errmsg, ier = curve_fit(model_func, unique_x, mean_y, p0=initial_guess,
                                                          full_output=True)
    return params,unique_x,mean_y,output_list_delta

def fit_3d(file_path):
    output_list = np.loadtxt(file_path)
    output_list1 = output_list[:, :2] - 17
    y = np.linalg.norm(output_list1, axis=1).reshape(-1, 1)
    idx = y > 0.01
    y = y[idx]
    x = np.zeros((18, 18, 18))
    for i in range(18):
        for j in range(18):
            for k in range(18):
                x[i, j, k] = np.linalg.norm(np.array((i + 8 - 17, j + 8 - 17)))
    x = x.reshape(-1, 1)
    unique_x, inverse_indices = np.unique(x[idx], return_inverse=True)
    mean_y = np.zeros(len(unique_x))
    for i in range(len(unique_x)):
        mask = (inverse_indices == i)
        mean_y[i] = y[mask].mean()
    initial_guess = [0.3, 8.4]
    params, covariance, infodict, errmsg, ier = curve_fit(model_func, unique_x, mean_y, p0=initial_guess,
                                                          full_output=True)
    return params,unique_x,mean_y

if __name__ == "__main__":
    mode = "fit_2d"
    if mode == 0:
        # generate_data_2d(save_path="output_list")
        generate_data_3d(save_path="output_list_3d")
    elif mode == "fit_2d":
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        file_path = "output_list"
        params, x, y,y_data = fit_2d(file_path)
        #0.211 , 9.50, 0.1,y_range(0,9)
        y_fit = model_func(x , params[0] , params[1])
        y_loss = (y_fit - y)
        fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=600)

        im = ax1.imshow(y_data.reshape(36, 36), cmap='viridis',
                        vmin=y_data.min(), vmax=y_data.max(),
                        origin='lower',
                        extent=[0, 36, 0, 36])

        ax1.set_title("Relationship between Center Shift Magnitude and Energy Injection Location",
                      fontsize=12, fontweight='bold', pad=10)

        ax1.set_xlabel('X Coordinate of Injection Location', fontsize=10)
        ax1.set_ylabel('Y Coordinate of Injection Location', fontsize=10)
        cbar = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Center Shift Magnitude', fontsize=10)
        ax1.set_xticks(np.arange(0, 37, 6))
        ax1.set_yticks(np.arange(0, 37, 6))

        plt.tight_layout()
        plt.savefig("D:/论文工作/小论文/res_plot/vt/2dmapping.png", dpi=600, bbox_inches='tight')
        plt.show()
        print(f"Max absolute loss: {np.max(abs(y_loss))}")
        print(f"Parameters: {params}")
        fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=600)

        ax2.plot(x, y,
                 label='Original Data',
                 color='#1f77b4', linestyle='-.', markersize=3, alpha=0.7)
        ax2.plot(x, y_fit,
                 label='Fitted Curve',
                 color='#d62728', linestyle='-', linewidth=2)
        ax2.set_title('Effect of Energy Injection Distance on Neuronal Center Shift',
                      fontsize=12, fontweight='bold', pad=10)

        ax2.set_xlabel('Energy Injection Distance', fontsize=10)
        ax2.set_ylabel('Neuronal Center Shift Distance', fontsize=10)
        ax2.legend(loc='best', fontsize=9, frameon=True)
        ax2.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig("D:/论文工作/小论文/res_plot/vt/fitted_curve.png", dpi=600, bbox_inches='tight')
        plt.show()
    else:
        file_path = "output_list_3d"
        params, x, y = fit_3d(file_path)
        print(params)
        y_fit = model_func(x, params[0], params[1])
        print(len(x))
        y_loss = y_fit - y
        print(np.max(abs(y_loss)))
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, y_fit)
        plt.show()

