from src.cann3d_in_math import CANN3D
from src.cann_in_math import CANN1D
from src.func import cann3d_data_save,cann1d_data_save
def _worker(q):
    cann = CANN3D(49)
    while True:
        task = q.get()         # (save_dir, filename, flatten)
        if task is None:
            break
        save_dir, filename, flatten = task
        cann3d_data_save(cann, 500, save_dir, filename, flatten)

def _worker1D(q):
    cann = CANN1D(49)
    while True:
        task = q.get()         # (save_dir, filename, flatten)
        if task is None:
            break
        save_dir, filename, flatten = task
        cann1d_data_save(cann, 500, save_dir, filename)