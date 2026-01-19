import os
from PIL import Image

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
            output_path = os.path.join(output_folder, filename)
            try:
                # Open the image file
                with Image.open(source_path) as img:
                    # Resize the image. Image.Resampling.LANCZOS is a high-quality resampling filter.
                    resized_img = img.resize(target_size, Image.LANCZOS)
                    # Save the resized image
                    resized_img.save(output_path)
                    print(f"Resized '{filename}' and saved to '{output_folder}'.")

            except Exception as e:
                print(f"Error processing '{filename}': {e}")

source_folder = "E:\\dataset\\dataset\\processed_data_2\\images\\"
output_folder = "E:\\dataset\\dataset\\processed_data_2\\images_320x180\\"
resize_and_save_images(source_folder, output_folder)


