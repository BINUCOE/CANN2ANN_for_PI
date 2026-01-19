import functools
import numpy as np
import matplotlib.pyplot as plt
from hdcn import yaw_height_hdc_network

def save_results_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        save_flag = kwargs.get('save_data', False)
        res_root = kwargs.get('res_root', "./default_res/")
        odoMap, expMap, GC_output, hdcn_output, gcn, mep, hdcn= func(*args, **kwargs)
        fig = plt.figure(figsize=(14, 10), dpi=100)
        ax1 = fig.add_subplot(131, projection='3d')
        odos = np.array(odoMap)[:, :3]
        x1 = odos[:, 0]
        y1 = odos[:, 1]
        z1 = odos[:, 2]
        ax1.scatter(x1 * 0.8, y1 * 0.8, z1 * 0.1)
        plt.title('Odometry')

        ax2 = fig.add_subplot(132, projection='3d')
        exps = np.array(expMap)[:, :3]
        x2 = exps[:, 0]
        y2 = exps[:, 1]
        z2 = exps[:, 2]
        ax2.scatter(x2 * 0.8, y2 * 0.8, z2 * 0.1)
        plt.title('Experience Map')

        ax3 = fig.add_subplot(133, projection='3d')
        gc_output = np.array(GC_output)[:, :3]
        x3 = gc_output[:, 0]
        y3 = gc_output[:, 1]
        z3 = gc_output[:, 2]
        ax3.scatter(y3 * 0.8, x3 * 0.8, z3 * 0.1)
        plt.title('GCoutput')
        plt.tight_layout()
        plt.tight_layout()
        if save_flag:
            print('saving....')
            np.savetxt(res_root + "odo_map.txt", odos)
            if type(hdcn) is yaw_height_hdc_network.YawHeightHDCNetwork:
                np.savetxt(res_root + "exp_math_map.txt", exps)
                plt.savefig(res_root + "tranj_math.jpg")
                np.savetxt(res_root + "math_hdcn_out.txt", np.array(hdcn_output))
                np.savetxt(res_root + "math_gcn_output.txt", gc_output)
            else:
                np.savetxt(res_root + "exp_ann_map.txt", exps)
                plt.savefig(res_root + "tranj_ann.jpg")
                # yaw_state = np.array(hdcn.yaw_state_history)
                # height_state = np.array(hdcn.height_state_history)
                # gcn_state = np.array(gcn.cann_ann_history)
                # frame_history = np.array(mep.frame_history).reshape(-1, 1)
                # np.savetxt(res_root+"yaw_state.txt",yaw_state)
                # np.savetxt(res_root+"height_state.txt", height_state)
                # np.savetxt(res_root+"gcn_state.txt", gcn_state)
                # np.savetxt(res_root + "frame_history", frame_history)
                np.savetxt(res_root+"hdcn_out.txt", np.array(hdcn_output))
                np.savetxt(res_root+"gcn_output.txt",gc_output)
                # vt_history = np.array(gcn.vt_history)
            print('all done')
        plt.show()
    return wrapper