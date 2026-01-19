import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from src.animation import CANN1d_Animation
from matplotlib import animation
from  src.func import decoder_in_math
from src.cann_in_math import CANN1D

def test_model():
    # 定义模型、以及输入流
    # vlaues与durations元素一一对应表示从0ms开始，每个输入值分别持续了多少秒，这里是2-12ms时输入为I1，其余为0,durations储存总时长
    cann2 = CANN1D(num=512, k=0.1)
    dur1, dur2, dur3, dur4 = 10., 20., 10., 10.
    num1 = int(dur1 / bm.get_dt())
    num2 = int(dur2 / bm.get_dt())
    num3 = int(dur3 / bm.get_dt())
    num4 = int(dur4 / bm.get_dt())
    I = bm.zeros(num1 + num2 + num3)
    I[num1:num1 + num2] = bm.linspace(0., 2 * bm.pi, num2)
    I[num1 + num2:num1 + num2 + num3] =  0.5 * bm.pi
    # Iext[0:num1] = 1
    # Iext[num1:num1+num2+num3] = bm.random.normal(0.,1. , (num2 + num3))
    I = I.reshape(-1, 1)
    Iext = cann2.get_stimulus_by_pos(I)
    duration = dur1 + dur2 + dur3
    noise = bm.random.normal(0.,1. , (int(duration/bm.get_dt()), len(cann2.x)))
    Iext += noise
    runner = bp.DSRunner(cann2, inputs=['input', Iext, 'iter'], monitors=['u'], dyn_vars=cann2.vars())
    runner.run(duration)
    u = runner.mon.u
    decoded_output = decoder_in_math(u)
    x = bm.arange(u.shape[0])
    I = bm.remainder(I , 2 * bm.pi)
    I = bm.where(I < bm.pi, I , I - 2 * bm.pi)
    plt.figure()
    plt.plot(x , decoded_output , label = 'decoded_output')
    plt.plot(x , I, label = 'Iext')
    plt.legend()
    plt.show()
    plt.figure()
    x = cann2.x
    plt.plot(x , noise[100])
    plt.plot(x , u[200])
    plt.show()
    # # show in ani
    # nums = int(duration / 2 / bm.get_dt())
    # x = cann2.x
    # cann_ani = CANN1d_Animation(Iext, runner.mon.u, x, )
    # # blit设置为false时会更新所有变量,但会保留之前的变量内容，需要清除画布
    # ani = animation.FuncAnimation(cann_ani.fig, func=cann_ani.fig_update, frames=nums, init_func=cann_ani.fig_init,
    #                               interval=40, blit=False)
    # # ani.save("行波随动-滞后性体现.gif", fps = 25,writer = 'imagemagick' )
    # plt.show()
    # print(bm.get_dt())