import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from scipy.linalg import orthogonal_procrustes

data_path = r'E:/dataset/dataset/processed_data'
gps_xyz = '/coords/gps_coords.txt'
image_path = data_path + '/images/'
image_xyz = "interpolated_image_poses.txt"
def load_image(image_path):
    img_list = []
    for img_name in os.listdir(image_path):
        img_list.append(image_path + img_name)

    img_list.sort()
    return img_list

def show_img(image_path, step):
    image_path_list = load_image(image_path)
    image_path_list = image_path_list[::step]
    # 3. 设置播放参数
    # 播放的帧率 (Frame Rate)，即每秒播放多少张图像
    fps = 60
    # 计算每帧的延迟（毫秒），1000 / fps
    delay_ms = int(1000 / fps*step)

    # 检查是否找到了图像
    if not image_path_list:
        print(f"在路径 '{image_path}' 中没有找到 .png 图像文件。请检查路径和文件格式。")
    else:
        print(f"找到了 {len(image_path_list)} 张图像，开始播放...")

        # 循环播放图像
        for image_path in image_path_list:
            # 读取图像
            frame = cv2.imread(image_path)

            # 检查图像是否读取成功
            if frame is None:
                print(f"警告：无法读取图像文件 {image_path}，已跳过。")
                continue

            # 在一个名为 "Image Player" 的窗口中显示图像
            cv2.imshow('Image Player', frame)

            # 等待指定的毫秒数，如果用户按下了 'q' 键则退出
            # cv2.waitKey() 返回按键的 ASCII 值
            key = cv2.waitKey(delay_ms)
            if key == ord('q'):
                print("用户按下了 'q' 键，退出播放。")
                break

        # 4. 释放资源
        # 销毁所有 OpenCV 窗口
        cv2.destroyAllWindows()
        print("播放完毕。")

def show_coords(coords_path):
    try:
        xyz = np.loadtxt(coords_path,delimiter=',')
    except ValueError:
        xyz = np.loadtxt(coords_path, delimiter=' ')
    plt.figure()
    plt.plot(50 * xyz[::5, 0],50 * xyz[::5, 1])
    plt.show()

def get_yaw():
    coords_path = 'ros_results/vo_xyz.txt'
    try:
        xyz = np.loadtxt(coords_path,delimiter=',')
    except ValueError:
        xyz = np.loadtxt(coords_path, delimiter=' ')
    x = xyz[: , 0]
    y = xyz[: , 1]
    yaw_gt = []
    x0 = x[0]
    y0 = x[0]
    for i in range(1 , xyz.shape[0]):
        x1 = x[i]
        y1 = y[i]
        theta = np.arctan2(x1 - x0, y1 - y0)
        yaw_gt.append(theta)
        x0 = x1
        y0 = y1
    yaw_gt = np.array(yaw_gt)
    np.savetxt("ros_results/yaw_vo.txt", yaw_gt)
def align_2d_points(points1, points2):
    # 检查输入点集是否有效
    if points1.shape != points2.shape:
        raise ValueError("Input point sets must have the same shape.")

    if points1.shape[0] != 2:
        raise ValueError("Input point sets must be 2D, with shape (2, N).")

    # 计算协方差矩阵 H
    # H = points1 @ points2.T
    H = np.dot(points1, points2.T)

    # 对协方差矩阵进行SVD分解
    U, _, V_T = np.linalg.svd(H)

    # 计算旋转矩阵 R
    # R = V @ U.T
    V = V_T.T
    R = np.dot(V, U.T)

    # 检查行列式，确保是纯旋转矩阵，而不是反射
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = np.dot(V, U.T)

    # 计算尺度因子 s
    # s = trace(R.T @ points1 @ points2.T) / trace(points2 @ points2.T)
    s = np.trace(np.dot(R.T, H)) / np.trace(np.dot(points2, points2.T))

    return R, s

def calculate_yaw_metrics(path_a, path_b):
    path_a = np.array(path_a)
    path_b = np.array(path_b)

    if path_a.shape != path_b.shape or path_a.shape[0] < 2:
        raise ValueError("路径形状必须相同且至少包含两个点。")

    N = path_a.shape[0]

    # 1. 计算方向向量
    vectors_a = path_a[1:] - path_a[:-1]
    vectors_b = path_b[1:] - path_b[:-1]

    angles_rad = []

    # 2. 遍历每对向量，计算夹角
    for v_a, v_b in zip(vectors_a, vectors_b):
        norm_a = np.linalg.norm(v_a)
        norm_b = np.linalg.norm(v_b)

        if norm_a == 0 or norm_b == 0:
            angle = 0.0
        else:
            dot_product = np.dot(v_a, v_b)
            cosine_angle = dot_product / (norm_a * norm_b)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)  # 结果在 [0, pi] 之间，即 [0, 180] 度

        angles_rad.append(angle)

    if not angles_rad:
        return 0.0, 0.0

    angles_rad = np.array(angles_rad)

    # 3. 计算最大偏航角
    max_angle_rad = np.max(angles_rad)
    max_angle_deg = np.degrees(max_angle_rad)

    # 4. 计算平均姿态角偏移 (算术平均值)
    # 因为 arccos 的结果已经在 [0, pi] 范围内，所以不需要取绝对值
    mean_angle_rad = np.mean(angles_rad)
    mean_angle_deg = np.degrees(mean_angle_rad)
    print(f"最大偏航角为： {max_angle_deg}")
    print(f"平均偏航角为： {mean_angle_deg}")
    return max_angle_deg, mean_angle_deg
def procrustes_alignment(X: np.ndarray, Y: np.ndarray, scale: bool = True):
    n, d = X.shape

    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)

    X_c = X - mu_X
    Y_c = Y - mu_Y

    H = Y_c.T @ X_c
    U, S, Vt = np.linalg.svd(H)

    R_optimal = U @ Vt

    R = R_optimal
    s_scipy = np.sum(S)

    if scale:
        norm_Y_c_sq = np.sum(Y_c ** 2)
        if norm_Y_c_sq == 0:
            s = 1.0
        else:
            s = s_scipy / norm_Y_c_sq
    else:
        s = 1.0

    t = mu_X

    Y_aligned = s * (Y_c @ R)
    Y_aligned = Y_aligned + mu_X
    dist1 = X[1: , :]- X[:-1, :]
    dist1 = np.linalg.norm(dist1, axis = 1)
    dist1 = np.sum(dist1)
    dist2 = Y_aligned[1: , :]- Y_aligned[:-1, :]
    dist2 = np.linalg.norm(dist2, axis = 1)
    dist2 = np.sum(dist2)
    print("轨迹1的长度为{:.4f}, 轨迹2的长度为{:.4f}".format(dist1, dist2))
    MAX_AE, MAE, RMSE= calculate_mae_rmse(Y_aligned, X)
    print("MAX_AE:{:.4f}, MAE:{:.4f}, RMSE{:.4f}".format(MAX_AE, MAE, RMSE))
    calculate_yaw_metrics(X, Y_aligned)
    return Y_aligned
def calculate_mae_rmse(points1, aligned_points2):
    errors = points1 - aligned_points2
    # 计算每个误差向量的模长
    # 欧几里得距离，即 sqrt(dx^2 + dy^2)
    distances = np.linalg.norm(errors, axis=1)
    max_ae = np.max(distances)
    # 计算MAE（平均绝对误差）
    mae = np.mean(distances)

    # 计算RMSE（均方根误差）
    rmse = np.sqrt(np.mean(distances ** 2))

    return max_ae , mae, rmse
if __name__ == "__main__":
    show_img("E:\\dataset\\dataset\\processed_data_clear\\images_320x180\\" , 4)
    # mep_ann = "../res_data_ros/exp_ann_map.txt"
    # gt = "interpolated_image_poses.txt"
    # frame = "../res_data_ros/frame_history"
    # gt = 200*np.loadtxt(gt)
    # frame = np.loadtxt(frame, dtype = np.integer)
    # frame = 5*np.squeeze(frame)
    # gt = gt[frame[1:-1], :2].T
    # mep_ann = np.loadtxt(mep_ann)
    # mep = mep_ann[1:, :2].T
    # aligned_points2 = procrustes_alignment(mep, gt)
    #
    # plt.figure()
    # plt.plot(mep[0,:], mep[1, :],label = "mep")
    # plt.plot(aligned_points2[0,:], aligned_points2[1, :],label = "GT")
    # plt.legend()
    # plt.show()
    # dist1 = mep[: , 1:]- mep[:, :-1]
    # dist1 = np.linalg.norm(dist1, axis = 0)
    # dist1 = np.sum(dist1)
    # dist2 = aligned_points2[: , 1:]- aligned_points2[:, :-1]
    # dist2 = np.linalg.norm(dist2, axis = 0)
    # dist2 = np.sum(dist2)
    # print("轨迹1的长度为{:.4f}, 轨迹2的长度为{:.4f}".format(dist1, dist2))
    # MAE, RMSE= calculate_mae_rmse(mep, aligned_points2)
    # print("MAE:{:.4f}, RMSE{:.4f}".format(MAE, RMSE))
    a = 0