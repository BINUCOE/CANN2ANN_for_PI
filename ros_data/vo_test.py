import visual_odometry_bak
import numpy as np
from run.main import load_image
from PIL import Image
import matplotlib.pyplot as plt

#data_0:FOV_HORI_DEGREE = 85, MAX_TRANS_V_THRESHOLD = 0.29,MAX_YAW_ROT_V_THRESHOLD = 3.4
image_path = "E:\\dataset\\dataset\\processed_data_clear\\images_320x180\\"
image_source_list = load_image(image_path)
vo = visual_odometry_bak.VisualOdometry(FOV_HORI_DEGREE=82.8, MAX_TRANS_V_THRESHOLD=0.76,
               MAX_YAW_ROT_V_THRESHOLD=4.0, ODO_TRANS_V_SCALE=30)

temp, odo_x, odo_y, odo_z = [np.pi / 2, 0.0, 0.0, 0.0]
startFrame = 0
endFrame = len(image_source_list)
# endFrame = 3000
n_steps = 2
odoMap = []
odo_yaw = []
flag = False
print(endFrame)
for i in range(startFrame, endFrame, n_steps):
    curImg = Image.open(image_source_list[i]).convert('L')

    # Visual templates and visual odometry use intensity, so convert to grayscale
    curGrayImg = np.clip(np.uint(curImg), 0, 255).astype(np.uint8)
    curGrayImg = np.float32(curGrayImg / 255.0)
    transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
    yawRotV *= np.pi / 180
    if not flag:
        flag = True
        odoMap.append([odo_x, odo_y, odo_z])
    else:
        # if 420 <= i <= 470:yawRotV = 0
        temp += yawRotV
        odo_x += transV * np.cos(temp)  # xcoord
        odo_y += transV * np.sin(temp)  # ycoord
        odo_z += heightV  # zcoord
        odoMap.append([odo_x, odo_y, odo_z])
        odo_yaw.append(temp*180/np.pi)
    show_IMG = np.clip(np.uint(curImg), 0, 255).astype(np.uint8)
    # cv2.imshow('Image Player',show_IMG[:90,:])
    # cv2.waitKey(2000)
    # print(temp*180/np.pi ,transV, i)
odos = np.array(odoMap)[:, :3]
np.savetxt("ros_results/vo_xyz.txt", odos)
x1 = odos[:, 0]
y1 = odos[:, 1]
z1 = odos[:, 2]
xyz_gt = np.loadtxt("interpolated_image_poses_clear.txt")
plt.figure()
plt.plot(x1, y1)
plt.plot( xyz_gt[:, 0], xyz_gt[:, 1])
plt.show()
np.savetxt("ros_results/odo_yaw.txt", odo_yaw)