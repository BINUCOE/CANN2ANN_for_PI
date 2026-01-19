import numpy as np
from scipy.interpolate import interp1d

def interpolate_poses(pose_timestamps_path, image_timestamps_path, pose_data_path):
    """
    Interpolates pose data to align with image timestamps.

    Args:
        pose_timestamps_path (str): Path to the text file containing pose timestamps.
        image_timestamps_path (str): Path to the text file containing image timestamps.
        pose_data_path (str): Path to the text file containing pose data (x, y, z, etc.).
    """
    try:
        # Load timestamps from files
        pose_timestamps = np.loadtxt(pose_timestamps_path, delimiter=',')
        image_timestamps = np.loadtxt(image_timestamps_path, delimiter=',')

        # Load pose data. Assumes pose data is a N x D matrix,
        # where N is the number of poses and D is the number of dimensions (e.g., x, y, z, roll, pitch, yaw)
        pose_data = np.loadtxt(pose_data_path, delimiter=',')

        # Check if the number of timestamps matches the number of pose data rows
        if len(pose_timestamps) != pose_data.shape[0]:
            print("Error: The number of pose timestamps does not match the number of pose data rows.")
            return

        # Create an interpolator for each dimension of the pose data
        # 'kind='linear'' specifies linear interpolation
        interpolators = []
        for i in range(pose_data.shape[1]):
            interpolator = interp1d(pose_timestamps, pose_data[:, i], kind='linear', fill_value='extrapolate')
            interpolators.append(interpolator)

        # Interpolate the pose data for each image timestamp
        interpolated_poses = []
        for t_img in image_timestamps:
            # Get the interpolated value for each dimension
            interp_pose = [interp(t_img) for interp in interpolators]
            interpolated_poses.append(interp_pose)

        # Convert the list of interpolated poses to a numpy array for easy handling
        interpolated_poses_array = np.array(interpolated_poses)

        # Save the interpolated poses to a new file
        output_path = 'interpolated_image_poses_2.txt'
        np.savetxt(output_path, interpolated_poses_array, fmt='%f', delimiter=' ')

        print(f"Successfully interpolated {len(interpolated_poses_array)} poses.")
        print(f"Output saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: One of the required files was not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

data_path = r'E:/dataset/dataset/processed_data_2'
pose_timestamp_path = '/timestamps/gps_timestamps.txt'
image_timestamp_path = '/timestamps/image_timestamps.txt'
pose_data = '/coords/gps_coords.txt'
interpolate_poses(
    pose_timestamps_path= data_path + pose_timestamp_path,
    image_timestamps_path= data_path + image_timestamp_path,
    pose_data_path= data_path + pose_data
)

image_pose = np.loadtxt("interpolated_image_poses_2.txt")
image_pose -= image_pose[0, :]
np.savetxt("interpolated_image_poses_2.txt", image_pose)