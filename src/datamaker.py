from multiprocessing import Process,Queue
from worker_logic import _worker,_worker1D
import gc


def start_worker(q):
    p = Process(target= _worker, args=(q,), daemon=True)
    p.start()
    return p
def datamaker():
    import jax
    #由于无法解决jax内存泄露问题，因此通过子进程的方式进行内存回收
    q = Queue()
    worker = start_worker(q)
    data_nums = 500
    save_dir = "../dataset_3d_49"
    RESTART_EVERY = 20
    for i in range(data_nums):
        filename = "cann3d_dict{:04d}.pkl".format(i)
        q.put((save_dir, filename, False))
        if (i + 1) % RESTART_EVERY == 0 or i == 2999:
            #每循环一定次数就进行重启子进程
            q.put(None)
            worker.join()
            worker.close()
            worker = start_worker(q)  # 启动新的 Worker
            gc.collect()
            jax.clear_caches()
            print(f"已重启 Worker，累计完成 {i + 1} 份数据")
    worker.join()
if __name__ == "__main__":
    datamaker()
