from PIL import Image
import numpy as np
import os
import time
import psutil
from pympler import asizeof
process = psutil.Process(os.getpid())


class NeuroSLAM:
    def __init__(self, image_source_list , ODO_STEP = 5):
        self.image_source_list = image_source_list
        self.DEGREE_TO_RADIAN = np.pi / 180
        self.idx_to_radian = np.pi / 18
        self.RADIAN_TO_DEGREE = 180 / np.pi
        self.KEY_POINT_SET = np.array([3750, 4700, 8193, 9210])
        self.ODO_STEP = ODO_STEP
        self.image_source_list = image_source_list[::self.ODO_STEP]
def load_image(image_path):
    img_list = []
    for img_name in os.listdir(image_path):
        img_list.append(image_path + img_name)
    
    return img_list

def rgb2gray(rgb_image):
    """
    Convert an RGB image to a grayscale image.

    Parameters:
    - rgb_image: A 3D numpy array representing the RGB image (shape: (height, width, 3))

    Returns:
    - gray_image: A 2D numpy array representing the grayscale image (shape: (height, width))
    """
    # Ensure the input is a numpy array
    rgb_image = np.array(rgb_image, dtype=np.float64)
    
    # Extract the R, G, and B channels
    R = rgb_image[:, :, 0]
    G = rgb_image[:, :, 1]
    B = rgb_image[:, :, 2]
    
    # Apply the grayscale conversion formula
    gray_image = (0.299 * R + 0.5870 * G + 0.1140 * B) / 255.0
    gray_image = np.clip(gray_image, 0, 1)
    
    return gray_image


def main(hdcn,gcn , vo, vt, mep, neuroslam,**kwargs):
    gcX, gcY, gcZ = gcn.get_gc_initial_pos()
    curYawTheta, curHeightValue = hdcn.get_hdc_initial_value()
    cpu_usages = []
    temp, odo_x, odo_y, odo_z = [0, 0.0, 0.0, 0.0]
    startFrame = 0
    image_source_list = neuroslam.image_source_list
    endFrame = len(image_source_list)
    # endFrame = 100
    curFrame = 0
    preImg = 0
    odoMap = []
    expMap = []
    GC_output = []
    hdcn_output = []
    cur_yaw_history = []
    odo_yaw = []
    t_start = time.time()
    for var_name, var_val in locals().copy().items():
        # 过滤掉内置函数和模块，只看用户定义的变量
        if not var_name.startswith('__'):
            size = asizeof.asizeof(var_val)
            if size > 1024:  # 只显示大于 1KB 的变量
                print(f"变量: {var_name:15} | 大小: {size / 1024 :7.2f} KB")
    for i in range(startFrame, endFrame):
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"\r[Iteration {i}] Memory Usage: {mem_mb:.2f} MB", end="")
        if i <= len(image_source_list):
            print(f"\nThe {i} / {endFrame} frame is processing ......")
        else:
            break
        curImg = Image.open(image_source_list[i]).convert('L')

        # Visual templates and visual odometry use intensity, so convert to grayscale
        curGrayImg = np.clip(np.uint(curImg), 0, 255).astype(np.uint8)
        curGrayImg = np.float32(curGrayImg / 255.0)

        # Computing the 3D odometry based on the current image
        # yawRotV in degree
        i *= neuroslam.ODO_STEP
        if len(neuroslam.KEY_POINT_SET) == 2:
            if i < neuroslam.KEY_POINT_SET[0]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
            elif i < neuroslam.KEY_POINT_SET[1]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 1)
            else:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 2)
            transV = 2.0
        elif len(neuroslam.KEY_POINT_SET) == 4:
            if i < neuroslam.KEY_POINT_SET[0]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
            elif i < neuroslam.KEY_POINT_SET[1]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 1)
            elif i < neuroslam.KEY_POINT_SET[2]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
            elif i < neuroslam.KEY_POINT_SET[3]:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 2)
            else:
                transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
        else:
            transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
        
        yawRotV *= neuroslam.DEGREE_TO_RADIAN  # in radian
        
        # Get the most active visual template
        curFrame += 1
        if vt.VT_STEP == 1:
            vtcurGrayImg = np.copy(curGrayImg)
        else:
            if np.mod(curFrame, vt.VT_STEP) == 1:
                vtcurGrayImg = np.copy(curGrayImg)
                preImg = vtcurGrayImg
            else:
                vtcurGrayImg = preImg
        
        vt_id, VT = vt.visual_template(vtcurGrayImg, gcX, gcY, gcZ, curYawTheta, curHeightValue)
        
        # Process the integration of yaw_height_hdc
        [curYawTheta, curHeightValue] = hdcn.yaw_height_hdc_iteration(vt_id, yawRotV, heightV, VT)
        hdcn_output.append([curYawTheta , curHeightValue])
        # curYawThetaInRadian = 0.  # Transform to radian
        curYawThetaInRadian = curYawTheta * hdcn.YAW_HEIGHT_HDC_Y_RES
        cur_yaw_history.append(curYawThetaInRadian)
        # 3D grid cells iteration
        [gcX, gcY, gcZ] = gcn.gc_iteration(vt_id, transV, curYawThetaInRadian, heightV, VT)
        GC_output.append([gcX, gcY, gcZ])

        # 3D experience map iteration
        mep.exp_map_iteration(vt_id, transV, yawRotV, heightV, gcX, gcY, gcZ, curYawTheta, curHeightValue, VT)
        # print(vt.NUM_VT, mep.NUM_EXPS,mep.CUR_EXP_ID,mep.EXPERIENCES[mep.CUR_EXP_ID].links)
        
        # Update PREV_VT_ID
        vt.PREV_VT_ID = vt_id
        mep.PREV_VT_ID = vt_id
        
        # For drawing visual odometry
        temp += yawRotV
        odo_x += transV * np.cos(temp)  # xcoord
        odo_y += transV * np.sin(temp)  # ycoord
        odo_z += heightV  # zcoord
        odoMap.append([odo_x, odo_y, odo_z])
        odo_yaw.append(temp)
        current_usage = process.cpu_percent(interval=None)
        cpu_usages.append(current_usage)

        # 计算统计结果
    max_load = max(cpu_usages)
    avg_load = sum(cpu_usages) / len(cpu_usages)
    print(f"最大负载: {max_load}%")
    print(f"平均负载: {avg_load:.2f}%")
    t_end = time.time()
    print("Time Consumption (s):", t_end - t_start)
    for ind in range(mep.NUM_EXPS):
        expTrajectory = [mep.EXPERIENCES[ind].x_exp, mep.EXPERIENCES[ind].y_exp, mep.EXPERIENCES[ind].z_exp]
        expMap.append(expTrajectory)
    for var_name, var_val in locals().copy().items():
        # 过滤掉内置函数和模块，只看用户定义的变量
        if not var_name.startswith('__'):
            size = asizeof.asizeof(var_val)
            if size > 1024:  # 只显示大于 1KB 的变量
                print(f"变量: {var_name:15} | 大小: {size / 1024:7.2f} KB")
    pass
    return odoMap, expMap, GC_output, hdcn_output, gcn, mep, hdcn





