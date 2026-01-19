import os
from PIL import Image
import numpy as np


def resize_and_save_images(source_folder, output_folder, target_size=(320, 180)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    # Supported image extensions
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    # Get a list of all files in the source folder
    file_list = os.listdir(source_folder)
    # Process each file in the folder
    for filename in file_list:
        # Check if the file is a supported image
        if filename.lower().endswith(supported_extensions):
            source_path = os.path.join(source_folder, filename)
            output_path = os.path.join(output_folder, "1"+filename)
            try:
                # Open the image file
                with Image.open(source_path) as img:
                    img.save(output_path)
                    print(f"Resized '{filename}' and saved to '{output_folder}'.")

            except Exception as e:
                print(f"Error processing '{filename}': {e}")

def test_resize_and_move():
    source_folder = "E:\\dataset\\dataset\\processed_data\\images\\"
    output_folder = "E:\\dataset\\dataset\\processed_data\\images_double\\"
    resize_and_save_images(source_folder, output_folder)


#concat the dataset2 and dataset1(2_end -> 1_start)
def test_concat_dataset():
    source_folder = "E:\\dataset\\dataset\\processed_data_1\\images_320x180"
    output_folder = "E:\\dataset\\dataset\\processed_data_concat\\images_320x180"
    souce_list = os.listdir(source_folder)
    target_list = os.listdir(output_folder)
    end_file_name = target_list[-1]
    for filename in souce_list:
        new_filename = int(end_file_name[:4]) + int(filename[:4])
        new_filename = str(new_filename) + '.png'
        souce_path = os.path.join(source_folder , filename)
        output_path = os.path.join(output_folder , new_filename)
        with Image.open(souce_path) as img:
              img.save(output_path)

def test_get_clear_poses():
    base_folder = 'E:\\dataset\\dataset\\processed_data_concat\\images_320x180'
    clear_folder = 'E:\\dataset\\dataset\\processed_data_clear\\images_320x180'
    souce_list = os.listdir(base_folder)
    target_list = os.listdir(clear_folder)
    org_poses = np.loadtxt("interpolated_image_poses_concat.txt")
    new_poses = np.zeros(shape = (len(target_list) , 3))
    j = 0
    for i in range(len(souce_list)):
        org_filename = os.path.join(base_folder , souce_list[i])
        cur_filename = os.path.join(clear_folder , target_list[j])
        with Image.open(org_filename) as img:
            org = np.array(img)
        with Image.open(cur_filename) as img:
            cur =  np.array(img)
        if (org == cur).all():
            new_poses[j] = org_poses[i]
            print(i ,j, len(target_list))
            j += 1

        if j == len(target_list):
            break
    new_poses -= new_poses[0]
    np.savetxt("interpolated_image_poses_clear.txt" , new_poses)
def test_concat_poses():
    pose1 = np.loadtxt("interpolated_image_poses_1.txt")
    pose2 = np.loadtxt("interpolated_image_poses_2.txt")
    pose1 = pose1 + pose2[-1]
    pose_concat = np.row_stack((pose2[:-1 , :] , pose1))
    np.savetxt("interpolated_image_poses_concat.txt" , pose_concat)
