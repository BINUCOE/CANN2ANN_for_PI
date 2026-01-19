import numpy as np
from src.cann_in_math import CANN1D
from src.func import decoder_in_math
import matplotlib.pyplot as plt
import brainpy.math as bm
import brainpy as bp
import time


def test_decoder_bais():
    cann1d = CANN1D(37,z_min=0,z_max=2*bm.pi)
    num_half = (cann1d.x.shape[0] -1 )/2
    Iext = bm.linspace(0 , 2*bm.pi , 10000).reshape(-1,1)
    Iext_encoded = cann1d.get_stimulus_by_pos(Iext)
    dur = Iext.shape[0]/10
    runner = bp.DSRunner(cann1d,inputs = ['input', Iext_encoded, 'iter'],monitors = ['u'], dyn_vars = cann1d.vars())
    runner.run(dur)
    x = np.arange(dur*10)
    u = runner.mon.u
    start_time = time.time()
    for i in range(1000):
        output = decoder_in_math(u,range = (0 , 2*bm.pi))
        # output = bm.where( output < 0, output + 2*bm.pi , output)
    end_time = time.time()
    print(end_time - start_time)
    start_time = time.time()
    for i in range(1000):
        output_rough1 = bm.argmax(u, dim=1,keepdim=True)*(bm.pi/num_half) - bm.pi
        # output_rough = bm.where(output_rough < 0, output_rough + 2 * bm.pi, output_rough)
        output_rough2 = bm.argmax(u, dim=1,keepdim=True)*(bm.pi/num_half) - bm.pi
        # output_rough = bm.where(output_rough < 0, output_rough + 2 * bm.pi, output_rough)
        output_rough3 = bm.argmax(u, dim=1,keepdim=True)*(bm.pi/num_half) - bm.pi
        # output_rough = bm.where(output_rough < 0, output_rough + 2 * bm.pi, output_rough)
        output_rough4 = bm.argmax(u, dim=1,keepdim=True)*(bm.pi/num_half) - bm.pi
        output_rough = output_rough4 + output_rough3 + output_rough1 + output_rough2
        # output_rough = bm.where(output_rough < 0, output_rough + 2 * bm.pi, output_rough)
    end_time = time.time()
    print(end_time - start_time)
    bais = bm.abs(Iext - output)
    max_bais = bm.max(bais)
    plt.legend()
    plt.plot(x , bais)
    plt.show()
    print("min_bais:{:.4f}".format(max_bais))
