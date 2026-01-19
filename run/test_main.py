from run.main import main,NeuroSLAM,load_image
from ros_data.visual_odometry_bak import VisualOdometry as VOROS
from ros_data.visual_template_bak import VisualTemplateManager as VTROS
from ros_data.multilayered_experience_map_bak import ExperienceMap as MEPROS
import multilayered_experience_map
import visual_odometry
import visual_template
import psutil
import os
process = psutil.Process(os.getpid())

image_path_dict = {'res_data_park/':"E:\\dataset\\neuroslam\\NeuroSLAM_Datasets\\01_NeuroSLAM_Datasets\\03_QUTCarparkData\\",
              'res_data_ros_clear/':"E:\\dataset\\dataset\\processed_data_clear\\images_320x180\\"}
save_data = False
from run.decorator import save_results_decorator
main = save_results_decorator(main)

def test_1():
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory Usage: {mem_mb:.2f} MB", end="")
    import torch
    import tensorflow as tf
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory Usage: {mem_mb:.2f} MB", end="")

def test_carpark_in_ann():
    from gcn import grid_cells_network_ann
    from hdcn import yaw_height_hdc_ann
    prefix_root = '../results/weights_49/'
    res_root = 'res_data_park/'
    weight1d_path = r"../temp/model1d_weights_49.pth"
    weight3d_path = r"../temp/model3d_weights_49.pth"
    nums = 49
    nums_1 = nums - 1
    image_path = image_path_dict[res_root]
    vo = visual_odometry.VisualOdometry()
    vt = visual_template.VisualTemplateManager()
    mep = multilayered_experience_map.ExperienceMap(GC_X_DIM = nums_1,GC_Y_DIM = nums_1,GC_Z_DIM = nums_1,
                                                    YAW_HEIGHT_HDC_Y_DIM = nums_1,YAW_HEIGHT_HDC_H_DIM = nums_1)
    # mep.loop_detection = False
    image_source_list = load_image(image_path)
    neuroslam = NeuroSLAM(image_source_list)
    hdc_ann = yaw_height_hdc_ann.Yaw_Height_HDC_ANN(nums , weight1d_path)
    gcn_ann = grid_cells_network_ann.GridCellNetwork(nums, weight3d_path)
    main(hdc_ann, gcn_ann, vo, vt, mep, neuroslam, res_root = prefix_root+res_root, save_data = save_data)

def test_carpark_in_ann_keras():
    from gcn import grid_cells_network_ann_keras
    from hdcn import yaw_height_hdc_ann_keras
    prefix_root = '../results/weights_49/'
    res_root = 'res_data_park/'
    weight1d_path = r"../temp/model1d_weights_49_keras.h5"
    weight3d_path = r"../temp/model3d_weights_49_keras.h5"
    nums = 49
    nums_1 = nums - 1
    image_path = image_path_dict[res_root]
    vo = visual_odometry.VisualOdometry()
    vt = visual_template.VisualTemplateManager()
    mep = multilayered_experience_map.ExperienceMap(GC_X_DIM = nums_1,GC_Y_DIM = nums_1,GC_Z_DIM = nums_1,
                                                    YAW_HEIGHT_HDC_Y_DIM = nums_1,YAW_HEIGHT_HDC_H_DIM = nums_1)
    # mep.loop_detection = False
    image_source_list = load_image(image_path)
    neuroslam = NeuroSLAM(image_source_list)
    hdc_ann = yaw_height_hdc_ann_keras.Yaw_Height_HDC_ANN(nums , weight1d_path)
    gcn_ann = grid_cells_network_ann_keras.GridCellNetwork(nums, weight3d_path)
    main(hdc_ann, gcn_ann, vo, vt, mep, neuroslam, res_root = prefix_root+res_root, save_data = save_data)

def test_carpark_in_math():
    from gcn import  grid_cells_network
    from hdcn import  yaw_height_hdc_network
    prefix_root = '../results/math/'
    res_root = 'res_data_park/'
    image_path = image_path_dict[res_root]
    vo = visual_odometry.VisualOdometry()
    vt = visual_template.VisualTemplateManager()
    mep = multilayered_experience_map.ExperienceMap()
    # mep.loop_detection = False
    image_source_list = load_image(image_path)
    neuroslam = NeuroSLAM(image_source_list)
    hdcn_math = yaw_height_hdc_network.YawHeightHDCNetwork()
    gcn_math = grid_cells_network.GridCellNetwork()
    main(hdcn_math, gcn_math, vo, vt, mep, neuroslam,res_root = prefix_root+res_root, save_data = save_data)

def test_rosdata_in_ann():
    from gcn import grid_cells_network_ann
    from hdcn import yaw_height_hdc_ann
    prefix_root = '../results/weights_25/'
    res_root = 'res_data_ros_clear/'
    weight1d_path = r"../temp/model1d_weights_37.pth"
    weight3d_path = r"../temp/model3d_weights_37.pth"
    nums = 37
    nums_1 = nums - 1
    vo = VOROS(FOV_HORI_DEGREE=82.8, MAX_TRANS_V_THRESHOLD=0.76,
               MAX_YAW_ROT_V_THRESHOLD=4.0, ODO_TRANS_V_SCALE=30)
    vt = VTROS()
    mep = MEPROS(EXP_LOOPS=5, EXP_CORRECTION=0.5,GC_X_DIM = nums_1,GC_Y_DIM = nums_1,GC_Z_DIM = nums_1,
                                                    YAW_HEIGHT_HDC_Y_DIM = nums_1,YAW_HEIGHT_HDC_H_DIM = nums_1)
    mep.loop_detection = True
    image_path = image_path_dict[res_root]
    image_source_list = load_image(image_path)
    neuroslam = NeuroSLAM(image_source_list, 2)
    neuroslam.KEY_POINT_SET = []
    hdc_ann = yaw_height_hdc_ann.Yaw_Height_HDC_ANN(nums , weight1d_path)
    gcn_ann = grid_cells_network_ann.GridCellNetwork(nums, weight3d_path)
    main(hdc_ann, gcn_ann, vo, vt, mep, neuroslam, res_root = prefix_root+res_root, save_data = save_data)


def test_rosdata_in_math():
    from gcn import  grid_cells_network
    from hdcn import  yaw_height_hdc_network
    res_root = 'res_data_ros_clear/'
    image_path = image_path_dict[res_root]
    image_source_list = load_image(image_path)
    neuroslam = NeuroSLAM(image_source_list ,2)
    neuroslam.KEY_POINT_SET = []
    vo = VOROS(FOV_HORI_DEGREE=82.8, MAX_TRANS_V_THRESHOLD=0.76,
               MAX_YAW_ROT_V_THRESHOLD=4.0, ODO_TRANS_V_SCALE=30)
    vt = VTROS()
    mep = MEPROS(EXP_LOOPS=5, EXP_CORRECTION=0.5)
    mep.loop_detection = True
    hdcn_math = yaw_height_hdc_network.YawHeightHDCNetwork()
    gcn_math = grid_cells_network.GridCellNetwork()
    main(hdcn_math, gcn_math, vo, vt, mep, neuroslam, res_root = res_root)


