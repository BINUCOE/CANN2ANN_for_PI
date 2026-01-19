import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
from matplotlib import cm
from ros_data.data_show import procrustes_alignment
from mpl_toolkits.mplot3d import Axes3D
class CANN_Animation:
    def __init__(self, X , cann_state, target,yaw_decoded, ylim = (0, 40)):
        self.U = cann_state
        self.X = X
        self.target = target
        self.ylim = ylim
        self.yaw_decoded = yaw_decoded
        self.fig, self.ax = plt.subplots()

    def fig_init(self , ):
        self.ax.set_ylim(self.ylim)
        line1, = self.ax.plot(self.X, self.U[0], "g" , label = "cann_state")
        line2 = self.ax.axvline(x = self.target[0], color='r', linestyle='-',label = "target")
        line3 = self.ax.axvline(x = self.yaw_decoded[0], color = "y", label="yaw_decoded")
        self.ax.set_title('t = {0:.1f}ms'.format(0))
        return line1,line2,line3

    def fig_update(self , i):
        self.ax.cla()
        self.ax.set_ylim(self.ylim)
        ts = i * 2
        line1, = self.ax.plot(self.X , self.U[ts] , "g" , label = "cann_state")
        line2 = self.ax.axvline(x = self.target[ts], color='r', linestyle='-',label = "target")
        line3 = self.ax.axvline(x = self.yaw_decoded[ts], color = "y", label="yaw_decoded")
        self.ax.set_title('t = {0:.1f}ms'.format(ts * 0.1))
        self.ax.legend()
        return line1,line2,line3

def get_delta(yaw_history):
    yaw_delta = yaw_history[1:] - yaw_history[:-1]
    yaw_delta = np.where(yaw_delta > np.pi, yaw_delta - 2 * np.pi, yaw_delta)
    yaw_delta = np.where(yaw_delta < -np.pi, yaw_delta + 2 * np.pi, yaw_delta)
    return yaw_delta
#文件路径列表，地图名列表
def show_map_xyz(file_list, map_name_list):
    map_list = []
    for file in file_list:
        map = np.loadtxt(file)
        map_list.append(map)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    x = map_list[0].shape[0]
    x = np.arange(x)
    for i in range(len(map_list)):
        axes[0].plot(x, map_list[i][:, 0], label = map_name_list[i] + "_X")
        axes[1].plot(x, map_list[i][:, 1], label = map_name_list[i] + "_Y")
        axes[2].plot(x, map_list[i][:, 2], label = map_name_list[i] + "_Z")
    plt.legend()
    plt.show()

def show_map_xy(file_list, map_name_list):
    map_list = []
    for file in file_list:
        map = np.loadtxt(file)
        if "poses" in file:
            map = 250 * map
            map = map[:, :2].T
        else:
            map = map[:,:2].T
        map_list.append(map)
    plt.figure(figsize = (10, 15))
    for i in range(len(map_list)):
        plt.scatter(map_list[i][0,:], map_list[i][1,:], label = map_name_list[i])
    plt.legend()
    plt.show()


def test_show_yaw():
    yaw_ann = np.loadtxt("results/cur_yaw_history_math.txt")
    yaw_ann = np.remainder(yaw_ann , 2*np.pi)
    yaw_gt = np.loadtxt("results/cur_yaw_history_ann.txt")
    yaw_gt = np.remainder(yaw_gt , 2*np.pi)
    x = np.arange(0 , yaw_ann.shape[0])
    plt.figure(figsize=(10 , 5))
    plt.plot(x , yaw_gt)
    plt.plot(x, yaw_ann)
    plt.legend()
    plt.show()

def test_show_exp():
    exp_map_list = ["./results/gc_output_ann.txt", "./results/gc_output_ann_keras.txt"]
    str_list = ["exp_ann_map", "exp_ann_keras_map"]
    show_map_xyz(exp_map_list, str_list)
    map1 = np.loadtxt(exp_map_list[0])
    map2 = np.loadtxt(exp_map_list[1])
    print(np.max(abs(map1 - map2)))

def test_show_rosmap():
    map_list = ["./ros_data/exp_math_map.txt", "./ros_data/exp_ann_map.txt","./ros_data/interpolated_image_poses.txt"]
    str_list = ["math_exp","ann", "gt"]
    show_map_xy(map_list, str_list)


#画头朝向细胞的波包状态
def test_show_hdcn():
    yaw_state = "res_data_park/yaw_state.txt"
    yaw_state = np.loadtxt( yaw_state)
    yaw_state = np.expand_dims(yaw_state,axis = 2)
    height_state = "res_data_park/height_state.txt"
    height_state = np.loadtxt(height_state)
    height_state = np.expand_dims(height_state,axis = 1)
    #up,plane,down
    # N_samples = [760 , 790,820,850]
    # N_samples= [1000 ,1020 , 1050 ,1100]
    N_samples = [1670,1690,1710,1730]
    fig_all = plt.figure(figsize=(20 , 5))  # 调整 figure 大小以容纳所有子图
    fig_all.subplots_adjust(wspace=0.3, hspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)
    for i in range(len(N_samples)):
        ax = fig_all.add_subplot(1, len(N_samples), i + 1, projection='3d')  # 1 行 N_samples 列

        # 获取指定索引的数据切片
        activity_data = yaw_state[N_samples[i]].dot(height_state[N_samples[i]])

        yaw_vals = np.arange(37)
        height_vals = np.arange(37)
        X, Y = np.meshgrid(yaw_vals, height_vals)
        x = np.arange(37)
        # plt.figure()
        # plt.plot(x,height_state[N_samples[i]].reshape(37))
        # plt.title(i)
        surf = ax.plot_surface(X, Y, activity_data,
                               cmap=cm.viridis,
                               linewidth=0,
                               antialiased=True)

        ax.set_xlabel('Height', labelpad=8,fontsize = 15)  # 调整 labelpad 避免重叠
        ax.set_ylabel('Yaw', labelpad=8,fontsize = 15)
        ax.set_zlabel('Activity', labelpad=0,fontsize = 15)
        ax.set_title(f"(c-{i + 1})", pad=-10,fontsize = 15)  # 标题在底部，并且调整 pad

        ax.set_zlim(0, 0.2)
        ax.view_init(elev=20, azim=-45)

        ticks = [0, 9, 18, 27, 36]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in [0, 9, 18, 27, 36]], fontsize=15)
        ax.set_yticklabels([str(t) for t in [0, 9, 18, 27, 36]], fontsize=15)
        ax.set_zticks([0, 0.1, 0.2])  # 限制 Z 轴刻度数量
        ax.set_zticklabels(['0', '0.1', '0.2'], fontsize=15)

        # 调整轴的颜色和线条，使其更接近您的图
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

    plt.tight_layout()
    plt.savefig("D:/论文工作/小论文/res_plot/cell_state/hdcn_down.png", dpi = 600)
    plt.show()
#画网格细胞的波包状态，画的时候为了效果明显可以固定x,y或者z的索引
def test_show_gcn():
    gcn = "res_data_park/gcn_state.txt"
    gcn = np.loadtxt(gcn)
    x = gcn[:,0:37].reshape(-1,37,1,1)
    y = gcn[:,37:74].reshape(-1,1,37,1)
    z = gcn[:,74:111].reshape(-1,1,1,37)

    #up,plane,down,画的时候可以把另外两轴索引固定，这样更直观[ 760，760，z_id],[x_id,1000,800]
    # N_samples = [760 , 790, 820, 850]
    # N_samples= [1000 ,1026 , 1030 ,1040]
    N_samples = [1640,1720,1800,2000]
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(left=0.08,    # 左侧留出更多空间给 Y 轴标签
                        right=0.90,   # 右侧留出更多空间给颜色条和右侧标签
                        bottom=0.2,  # 底部留出更多空间给 X 轴标签
                        top=0.95,
                        wspace=0.3,   # 子图之间的横向间距
                        hspace=0.1)
    from skimage.measure import marching_cubes
    for i in range(len(N_samples)):
        id = N_samples[i]
        # 1. 提取三维切片 (A[n, :, :, :])
        data_3d = x[id]*y[1000]*z[870]
        # 2. 创建 3D 子图 (1行5列，第i+1个)
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        # --- 核心可视化部分：绘制等值面 ---
        # 定义等值面阈值 (需要根据您的数据特性来定)
        threshold = 0.015

        # 使用 Marching Cubes 算法获取等值面顶点和面
        try:
            # 注意: marching_cubes可能需要调整参数以适应您的坐标系统
            verts, faces, _, _ = marching_cubes(data_3d, level=threshold)
            # 绘制等值面 (例如，蓝色/紫色团块)
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                            color='blue', alpha=0.6, antialiased=False)
        except Exception as e:
            # 如果等值面为空或出错，跳过绘制
            print(f"Sampling index {i} failed to generate isosurface: {e}")
        # --- 3. 设置坐标轴和标题 ---
        ax.set_xlabel('x',fontsize = 15)
        ax.set_ylabel('y',fontsize = 15)
        ax.set_zlabel('z',fontsize = 15)

        # 确保坐标轴范围统一
        ax.set_xlim(0, 36)
        ax.set_ylim(0, 36)
        ax.set_zlim(0, 27)

        # 设置刻度（例如 0, 9, 18, 27, 36）
        # 注意：图中 y 轴和 z 轴的刻度是反的，需要根据您的习惯调整
        ax.set_xticks(np.arange(0, 37, 9))
        ax.set_yticks(np.arange(0, 37, 9))
        ax.set_zticks(np.arange(0, 30, 9))
        ax.tick_params(axis='both',
                       labelsize=12)
        ax.grid(False)
        # 设置子图标题
        ax.set_title(f'(a-{i + 1})', y=-0.3,fontsize = 15)
    plt.savefig("D:/论文工作/小论文/res_plot/cell_state/gcn_plane.png",dpi = 600)
    plt.show()
#画park数据集下的轨迹以及经验地图
def test_park_plot_vo_mep():
    vo_data = "results/weights_25/res_data_park/odo_map.txt"
    mep_math = "results/weights_25/res_data_park/exp_math_map.txt"
    mep_ann = "results/weights_49/res_data_park//exp_ann_map.txt"
    vo_data = np.loadtxt(vo_data)
    mep_math = np.loadtxt(mep_math)
    mep_ann = np.loadtxt(mep_ann)
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(1, 4, 1, projection = '3d')
    fig.subplots_adjust(left=0.08,    # 左侧留出更多空间给 Y 轴标签
                        right=0.85,   # 右侧留出更多空间给颜色条和右侧标签
                        bottom=0.2,  # 底部留出更多空间给 X 轴标签
                        top=0.95,
                        wspace=0.3,   # 子图之间的横向间距
                        hspace=0.1)
    ax.view_init(elev=30, azim=210)
    ax.scatter(vo_data[:,0] * 0.8, vo_data[:,1]* 0.8, vo_data[:,2] * 0.1, s = 2.0)
    ax.set_title("视觉里程计",fontsize = 15)
    ax.set_xlabel('x (m)',fontsize = 15)
    ax.set_ylabel('y (m)',fontsize = 15)
    ax.set_zlabel('z (m)',fontsize = 15)
    ax.tick_params(axis='both',
                    labelsize=15)
    ax.set_xticks(np.arange(-20,31, 20))
    ax.set_yticks(np.arange(-20,51, 20))
    ax.set_zticks(np.arange(0, 3.1,1))
    ax.set_box_aspect([36,36,27])
    ax = fig.add_subplot(1, 4, 2, projection = '3d')
    ax.view_init(elev=30, azim=210)
    ax.scatter(mep_math[:,0] * 0.8, mep_math[:,1]* 0.8, mep_math[:,2] * 0.1,s = 2.0)
    ax.set_title("原算法",fontsize = 15)
    ax.set_xlabel('x (m)', fontsize=15)
    ax.set_ylabel('y (m)', fontsize=15)
    ax.set_zlabel('z (m)', fontsize=15)
    ax.tick_params(axis='both',
                   labelsize=15)
    ax.set_box_aspect([36,36,27])
    ax.set_xticks(np.arange(-10,40, 20))
    ax.set_yticks(np.arange(-10,51, 20))
    ax.set_zticks(np.arange(0, 3.1, 1))
    ax = fig.add_subplot(1, 4, 3, projection = '3d')
    ax.view_init(elev=30, azim=210)
    ax.scatter(mep_ann[:,0] * 0.8, mep_ann[:,1]* 0.8, mep_ann[:,2] * 0.1,color = 'r',s = 2.0)
    ax.set_title("本发明",fontsize = 15)
    ax.set_xlabel('x (m)', fontsize=15)
    ax.set_ylabel('y (m)', fontsize=15)
    ax.set_zlabel('z (m)', fontsize=15)
    ax.tick_params(axis='both',
                   labelsize=15)
    ax.set_box_aspect([36,36,27])
    ax.set_xticks(np.arange(-10,40, 20))
    ax.set_yticks(np.arange(-10,51, 20))
    ax.set_zticks(np.arange(0, 3.1, 1))
    ax = fig.add_subplot(1, 4, 4)
    ax.scatter(mep_math[::2 ,0] * 0.8, mep_math[::2,1]* 0.8,color = 'r',label = '原算法',s=20, marker = '+')
    ax.scatter(mep_ann[::2 , 0] * 0.8, mep_ann[::2 , 1] * 0.8, color = 'b', label = '本发明',s=20, marker='x')
    ax.set_title("经验地图比对",fontsize = 15)
    ax.set_xlabel('x (m)', fontsize=15)
    ax.set_ylabel('y (m)', fontsize=15)
    ax.tick_params(axis='both',
                   labelsize=15)
    ax.set_box_aspect(1.0)
    ax.set_xticks(np.arange(-10,40, 10))
    ax.set_yticks(np.arange(-10,51, 10))
    ax.legend(
        loc='upper left',  # 将图例的左上角作为锚点
        bbox_to_anchor=(1.02, 1),  # 将图例放置在绘图区域的右上方，超出边界
        borderaxespad=0.,  # 图例和轴线之间的填充
        fontsize=15,  # 调整字体大小，例如 'small', 'medium', 'large' 或具体数值
        frameon=True,  # 显示图例边框
        facecolor='white',  # 设置图例背景颜色
        edgecolor='gray',  # 设置图例边框颜色
        ncol=1  # 如果图例项多，可以设为2或更多列
    )
    fig.savefig("D:/论文工作/小论文/res_plot/map/park_data.png",dpi=600)
    plt.show()
#画park数据集下的细胞解码值
def test_park_plot_deocoded_data():
    ann_gc_out = "res_data_park/gcn_output.txt"
    math_gc_out = "res_data_park/math_gcn_output.txt"
    ann_hdc_out = "res_data_park/hdcn_out.txt"
    math_hdcn_out = 'res_data_park/math_hdcn_out.txt'
    ann_gc_out = np.loadtxt(ann_gc_out)
    ann_hdc_out = np.loadtxt(ann_hdc_out)
    math_gc_out = np.loadtxt(math_gc_out)
    math_hdcn_out = np.loadtxt(math_hdcn_out)
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(left=0.08,    # 左侧留出更多空间给 Y 轴标签
                        right=0.90,   # 右侧留出更多空间给颜色条和右侧标签
                        bottom=0.2,  # 底部留出更多空间给 X 轴标签
                        top=0.95,
                        wspace=0.3,   # 子图之间的横向间距
                        hspace=0.1)
    ax = fig.add_subplot(1,4,1,projection = '3d')
    ax.view_init(elev=30, azim=210)
    ax.scatter(ann_gc_out[:,0]*0.8, ann_gc_out[:,1]*0.8, ann_gc_out[:,2] * 0.1,s = 2.0)
    ax.set_title("GC Output in Reconstructed NeuroSLAM",fontsize = 12)
    ax.set_xlabel('x',fontsize = 12)
    ax.set_ylabel('y',fontsize = 12)
    ax.set_zlabel('z',fontsize = 12)
    ax.set_box_aspect([36,36,27])
    ax.set_xticks(np.arange(0,40, 10))
    ax.set_yticks(np.arange(0,40, 10))
    ax.set_zticks(np.arange(0, 4, 1))
    ax = fig.add_subplot(1,4,2,projection = '3d')
    ax.view_init(elev=30, azim=210)
    ax.scatter(math_gc_out[:,0]*0.8, math_gc_out[:,1]*0.8, math_gc_out[:,2] * 0.1, s = 2.0)
    ax.set_title("GC Output in NeuroSLAM",fontsize = 12)
    ax.set_xlabel('x',fontsize = 12)
    ax.set_ylabel('y',fontsize = 12)
    ax.set_zlabel('z',fontsize = 12)
    ax.set_box_aspect([36,36,27])
    ax.set_xticks(np.arange(0,40, 10))
    ax.set_yticks(np.arange(0,40, 10))
    ax.set_zticks(np.arange(0, 4, 1))
    ax = fig.add_subplot(1,4,3)
    ax.plot(ann_hdc_out[:,0], ann_hdc_out[:,1])
    ax.set_title("HDC Output in Reconstructed NeuroSLAM",fontsize = 12)
    ax.set_xlabel('yaw',fontsize = 12)
    ax.set_ylabel('height',fontsize = 12)
    ax.set_xticks(np.arange(0,37, 6))
    ax.set_yticks(np.arange(0,37, 6))
    ax.set_box_aspect(1.0)
    ax = fig.add_subplot(1,4,4)
    ax.plot(math_hdcn_out[:,0], math_hdcn_out[:,1])
    ax.set_title("HDC Output in Origin NeuroSLAM",fontsize = 12)
    ax.set_xlabel('yaw',fontsize = 12)
    ax.set_ylabel('height',fontsize =12)
    ax.set_xticks(np.arange(0,37, 6))
    ax.set_yticks(np.arange(0,37, 6))
    ax.set_box_aspect(1.0)
    plt.savefig("D:/论文工作/小论文/res_plot/cell_state/park.png",dpi = 600)
    plt.show()
#画ros数据集下的轨迹以及经验地图
def test_ros_plot_vo_mep():
    vo_data = "results/weights_37/res_data_ros_clear/odo_map.txt"
    mep_math = "results/weights_37/res_data_ros_clear/exp_math_map.txt"
    mep_ann = "results/weights_25 /res_data_ros_clear/exp_ann_map.txt"
    gt = "ros_data/interpolated_image_poses_clear.txt"
    frame = "results/weights_37/res_data_ros_clear/frame_history"
    vo_data = np.loadtxt(vo_data)
    # plt.plot(orb[:,0],orb[:,1])
    # plt.show()
    mep_math = np.loadtxt(mep_math)
    mep_ann = np.loadtxt(mep_ann)
    gt = np.loadtxt(gt)
    frame = np.loadtxt(frame, dtype=np.integer)
    frame_true = 2 * np.squeeze(frame)
    gt = gt[frame_true[1:-1], :2]
    mep_ann = mep_ann[1:, :2]
    mep_ann = procrustes_alignment(gt, mep_ann)
    mep_math= mep_math[1:, :2]
    print(gt.shape , mep_ann.shape , mep_math.shape)
    mep_math = procrustes_alignment(gt, mep_math)

    # orb = orb[frame_true[1:-1]]
    # orb = procrustes_alignment(gt,orb)
    vo_data = vo_data[frame[1:-1], :2]
    vo_data = procrustes_alignment(gt, vo_data)
    fig = plt.figure(figsize=(20 ,5))
    ax = fig.add_subplot(1, 4, 1)
    fig.subplots_adjust(left=0.08,  # 左侧留出更多空间给 Y 轴标签
                        right=0.85,  # 右侧留出更多空间给颜色条和右侧标签
                        bottom=0.2,  # 底部留出更多空间给 X 轴标签
                        top=0.95,
                        wspace=0.3,  # 子图之间的横向间距
                        hspace=0.1)
    ax.scatter(vo_data[::2, 0], vo_data[::2, 1], color='y',s= 10)
    ax.set_title("视觉里程计", fontsize=15)
    ax.set_xlabel('x (m)',fontsize = 15)
    ax.set_ylabel('y (m)', fontsize = 15)
    ax.set_box_aspect(1.0)
    ax.set_xticks(np.arange(-30, 11, 5))
    ax.set_yticks(np.arange(-15, 26, 5))
    ax.tick_params(axis='both',
                   labelsize=15)
    ax = fig.add_subplot(1, 4, 2)
    ax.scatter(mep_math[::2, 0], mep_math[::2, 1], color='r',s = 10)
    ax.set_title("原算法", fontsize=15)
    ax.set_xlabel('x (m)',fontsize = 15)
    ax.set_ylabel('y (m)', fontsize = 15)
    ax.set_box_aspect(1.0)
    ax.set_xticks(np.arange(-30, 11, 5))
    ax.set_yticks(np.arange(-15, 26, 5))
    ax.tick_params(axis='both',
                   labelsize=15)
    ax = fig.add_subplot(1,4,3)
    ax.set_title("本发明", fontsize=15)
    ax.scatter(mep_ann[::2, 0], mep_ann[::2, 1], color='b', s =10)
    ax.set_xlabel('x (m)',fontsize = 15)
    ax.set_ylabel('y (m)', fontsize = 15)
    ax.set_box_aspect(1.0)
    ax.set_xticks(np.arange(-30, 11, 5))
    ax.set_yticks(np.arange(-15, 26, 5))
    ax.tick_params(axis='both',
                   labelsize=15)
    ax = fig.add_subplot(1,4,4)
    ax.scatter(gt[::2, 0], gt[::2, 1], color='g', label='真值', s=10)
    # ax.scatter(vo_data[::2, 0], vo_data[::2, 1], color='y', label='Pure VO', s=20, marker='1')
    ax.scatter(mep_math[::2,0], mep_math[::2 , 1], color='r', label='原算法', s=20,marker='+')
    # ax.scatter(orb[::2, 0], orb[::2, 1], color='c', label='ORB-SLAM2', s=20, marker='2')
    ax.scatter(mep_ann[::2,0], mep_ann[::2 , 1], color='b', label='本发明', s=20,marker='x')
    ax.set_title("轨迹比对",fontsize = 15)
    ax.set_xlabel('x (m)',fontsize = 15)
    ax.set_ylabel('y (m)', fontsize = 15)
    ax.set_box_aspect(1.0)
    ax.set_xticks(np.arange(-30, 11, 5))
    ax.set_yticks(np.arange(-15, 26, 5))
    ax.tick_params(axis='both',
                   labelsize=15)
    plt.legend(
        loc='upper left',  # 将图例的左上角作为锚点
        bbox_to_anchor=(1.02, 1),  # 将图例放置在绘图区域的右上方，超出边界
        borderaxespad=0.,  # 图例和轴线之间的填充
        fontsize=15,  # 调整字体大小，例如 'small', 'medium', 'large' 或具体数值
        frameon=True,  # 显示图例边框
        facecolor='white',  # 设置图例背景颜色
        edgecolor='gray',  # 设置图例边框颜色
        ncol=1
    )
    # plt.savefig("D:/论文工作/小论文/res_plot/map/ros_data.png",dpi = 600)
    plt.show()

def test_show_hdcn_output():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    hdcn_with_LCD = np.loadtxt('results/math/res_data_park/math_hdcn_out.txt')
    hdcn_without_LCD = np.loadtxt('results/math_no_LCD/res_data_park/math_hdcn_out.txt')
    x = np.arange(hdcn_with_LCD.shape[0])
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(x, hdcn_without_LCD[:, 0],
            label='HDCN without LCD',
            color='#1f77b4', linestyle='--', linewidth=1.5)

    ax.plot(x, hdcn_with_LCD[:, 0],
            label='HDCN with LCD',
            color='#d62728', linestyle='-', linewidth=1.5)
    ax.set_title('Comparison of Decoding Values', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Sample Index (or Time)', fontsize=12)
    ax.set_ylabel('Decoding Value', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig('results/decoding_comparison.jpg', format='jpg', bbox_inches='tight')
    plt.show()
    print(np.abs(hdcn_with_LCD[-1,0] - hdcn_without_LCD[-1,0]))




