from scipy.optimize import minimize
import visual_odometry_bak
import numpy as np
from test.main import load_image
from PIL import Image
from data_show import align_2d_points


def run_vo_with_op(params):
    image_path = "E:\\dataset\\dataset\\processed_data_clear\\images_320x180\\"
    image_source_list = load_image(image_path)
    vo = visual_odometry_bak.VisualOdometry(FOV_HORI_DEGREE = params[1], MAX_TRANS_V_THRESHOLD = params[0],
                                            MAX_YAW_ROT_V_THRESHOLD = params[2])

    temp, odo_x, odo_y, odo_z = [np.pi / 2, 0.0, 0.0, 0.0]
    startFrame = 0
    endFrame = len(image_source_list)
    n_steps = 2
    odoMap = []
    odo_yaw = []
    flag = False
    for i in range(startFrame, endFrame, n_steps):
        curImg = Image.open(load_image(image_path)[i]).convert('L')

        # Visual templates and visual odometry use intensity, so convert to grayscale
        curGrayImg = np.clip(np.uint(curImg), 0, 255).astype(np.uint8)
        curGrayImg = np.float32(curGrayImg / 255.0)
        transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
        yawRotV *= np.pi / 180
        if not flag:
            flag = True
            odoMap.append([odo_x, odo_y, odo_z])
        else:
            temp += yawRotV
            odo_x += transV * np.cos(temp)  # xcoord
            odo_y += transV * np.sin(temp)  # ycoord
            odo_z += heightV  # zcoord
            odoMap.append([odo_x, odo_y, odo_z])
            odo_yaw.append(yawRotV)
    odos = np.array(odoMap)[:, :2]
    return odos
def cal_ATE(T_est, T_gt):
    R, s = align_2d_points(T_est , T_gt)
    T_gt = s*R.dot(T_gt)
    errors = T_est - T_gt
    distances = np.linalg.norm(errors, axis=0)
    ATE = np.sum(distances)

    return ATE

def trajectory_cost_function(params):
    T_est = run_vo_with_op(params)
    T_gt = np.loadtxt("interpolated_image_poses_clear.txt")
    T_gt = T_gt[::2, :2].T
    print(T_est.shape , T_gt.shape)
    # 步骤 2: 计算 ATE
    # 这一步需要先进行相似变换对齐 (例如使用 umeyama 算法)
    ATE = cal_ATE(T_est.T, T_gt)
    print(f"当前ATE为: {ATE},params:{params}")
    return ATE


x0 = np.array([0.72, 75, 4.2]) # 初始 FOV_H, FOV_V
bounds = [(0.6, 1.4),(60, 90),(1 , 10)]
# 2. 运行优化
# 使用 'Powell' 方法，因为它在黑箱优化中表现优秀
result = minimize(
    fun=trajectory_cost_function,
    x0=x0,
    method='Powell',
    bounds=bounds,
    options={'maxfev':200, 'ftol': 1e-4} # maxfev: 最大函数调用次数 (防止运行时间过长)
)

min_ate = result.fun

print(f"优化结果：{'成功' if result.success else '失败'}")
print(result.x)
print(f"最小 ATE: {min_ate:.4f}")