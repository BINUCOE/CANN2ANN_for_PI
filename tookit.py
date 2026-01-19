import numpy as np
import os


def rename_files_by_timestamp(f_path):
    """
    Rename all .txt files in the specified folder by their timestamp order.

    Parameters:
    - folder_path: Path to the folder containing the .txt files.
    """
    files = [f for f in os.listdir(f_path) if f.endswith('.txt')]

    files.sort()
    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(f_path, filename)
        new_path = os.path.join(f_path, f"{index}.txt")
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{index}.txt'")


if __name__ == "__main__":
    folder_path = ".\\data_test\\vo\\"
    rename_files_by_timestamp(folder_path)

