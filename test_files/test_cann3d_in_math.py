import brainpy.math as bm
import brainpy as bp
from src.cann3d_in_math import CANN3D
from src.func import cann3d_decoder_in_math
from src.animation import CANN3d_Animation
from matplotlib import animation

def test_model():
    cann2 = CANN3D(num=36, k=0.1)
    dur1, dur2, dur3, dur4 = 10., 20., 20., 20.
    num1 = int(dur1 / bm.get_dt())
    num2 = int(dur2 / bm.get_dt())
    num3 = int(dur3 / bm.get_dt())
    num4 = int(dur4 / bm.get_dt())
    duration = dur1 + dur2 + dur3 + dur4
    Iext = bm.zeros((num1 + num2 + num3 + num4, 3))
    Iext[num1:num1 + num2, 0] = bm.linspace(0., 2 * bm.pi, num2)
    Iext[num1 + num2:, 0] = 2 * bm.pi
    Iext[num1 + num2:num1 + num2 + num3, 1] = bm.linspace(0., 2 * bm.pi, num3)
    Iext[num1 + num2 + num3:, 1] = 2 * bm.pi
    Iext[num1 + num2 + num3:, 2] = bm.linspace(0., 2 * bm.pi, num4)
    # (times , xlim ,ylim ,zlim)
    Iext_encoded = cann2.get_stimulus_by_pos(Iext)
    runner = bp.DSRunner(cann2, inputs=['input', Iext_encoded, 'iter'], monitors=['u'], dyn_vars=cann2.vars())
    runner.run(duration)
    decoded_output = cann3d_decoder_in_math(runner.mon.u)
    # (times,xlim ,ylim)
    Iext_encoded = bm.sum(Iext_encoded, axis=3)
    nums = int(duration / 2)
    runner.mon.u = bm.sum(runner.mon.u, axis=3)
    cann_ani = CANN3d_Animation(Iext_encoded, runner.mon.u, Iext, decoded_output)
    # blit设置为false时会更新所有变量,但会保留之前的变量内容，需要清除画布
    ani = animation.FuncAnimation(cann_ani.fig, func=cann_ani.fig_update, frames=nums, init_func=cann_ani.fig_init,
                                  interval=40, blit=False)
    ani.save("3dcann_in_math.gif", fps=25, writer='imagemagick')