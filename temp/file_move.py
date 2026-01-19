import os
import shutil
from test.main import load_image
from pathlib import Path

image_path = "E:\\dataset\\neuroslam\\NeuroSLAM_Datasets\\01_NeuroSLAM_Datasets\\03_QUTCarparkData\\"
image_source_list = load_image(image_path)
target_folder = Path("E:\\dataset\\neuroslam\\NeuroSLAM_Datasets\\01_NeuroSLAM_Datasets\\04_QUTCarparkData_smapled\\")
# 采样
sampled_files = image_source_list[::5]
target_folder.mkdir(parents=True, exist_ok=True)
# 移动文件
for f in sampled_files:
    filename = os.path.basename(f)
    shutil.move(f, target_folder / filename)

print(f"已采样并移动 {len(sampled_files)} 张照片到 {target_folder}")