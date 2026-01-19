import brainpy as bp
import brainpy.math as bm

# 这里将网格细胞构建为一个正立方体模型，只规定一个维度的range
class CANN3D(bp.NeuGroup):
    def __init__(self, num, tau=0.1, k=8.1, a=0.5, A=10, J0=0.4, z_min=-bm.pi, z_max=bm.pi, **kwargs):
        # 调用父类构造函数
        super(CANN3D, self).__init__(size=num, **kwargs)
        # 参数初始化
        self.tau = tau
        self.k = k
        self.a = a
        self.A = A
        self.J0 = J0
        self.num = num
        self.num_3 = num * num * num
        # 特征空间相关参数初始化
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, num)
        self.rho = num / self.z_range
        self.dx = self.z_range / num
        # 初始化变量
        self.u = bm.Variable(bm.zeros((num ,num, num)))  # bm.Variable声明动态变量，使得即使该变量转入JIT编译器也依旧可随时间动态变化
        self.input = bm.Variable(bm.zeros((num,num,num)))
        # 利用成员函数生成连接矩阵
        self.conn_mat = self.make_conn(self.x)
        self.conn_mat = bm.fft.fftn(bm.fft.ifftshift(self.conn_mat))
        # 定义积分函数
        self.integral = bp.odeint(self.derivative)  # bp.odient()为ode积分器，用于计算微分方程组各变量的时变结果

    # 微分方程组
    def derivative(self, u, t, Iext):
        u2 = bm.square(u)
        r = u2 / (1+self.k * self.rho * bm.sum(u2))
        #FFT to complete the periodic computation
        r_freq = bm.fft.fftn(r)
        conv_freq = r_freq * self.conn_mat
        Irec = bm.fft.ifftn(conv_freq).real
        du = (-u + Irec + Iext) / self.tau
        return du

    # 将距离转换到[-z_range/2 , z_range/2]
    def dist(self, d):
        d = bm.remainder(d, self.z_range)  # 取余，同时保证结果与range具有同符号，这样操作使得d始终小于z_range
        # 该语句使得在计算距离时总是返回绝对值最小的距离，也符合圆上任意两点之间的小圆弧低于一半圆周
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range,
                     d)  # bm.where(condition,x,y)=condition?x:y,这样将d直接映射到-range/2到range/2中
        return d

    # 生成连接矩阵
    def make_conn(self, x):
        x_left= x.reshape(-1 , 1)
        d = self.dist(x_left - x_left[0])
        # 先算出1DCANN下的连接矩阵，在此基础上实现3DCANN的连接矩阵
        jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
        jxx = jxx.reshape(self.num , 1 , 1) * jxx.reshape(1 , self.num , 1) * jxx.reshape(1 , 1, self.num)
        return jxx

    # 为Iext
    def get_stimulus_by_pos(self, pos):
        x_input = bm.exp(-0.25 * bm.square(self.dist(self.x - pos[: ,0].reshape(-1,1)) / self.a))
        y_input = bm.exp(-0.25 * bm.square(self.dist(self.x - pos[: ,1].reshape(-1,1)) / self.a))
        z_input = bm.exp(-0.25 * bm.square(self.dist(self.x - pos[: ,2].reshape(-1,1)) / self.a))
        input = x_input.reshape(pos.shape[0],self.num,1,1) * y_input.reshape(pos.shape[0],1, self.num, 1) * z_input.reshape(pos.shape[0],1, 1, self.num)
        return self.A * input

    def update(self):
        self.u[:] = self.integral(self.u, bp.share['t'], self.input, bp.share['dt'])
        self.input[:] = 0.  # 重置外部输入


