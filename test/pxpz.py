import numpy as np
import matplotlib.pyplot as plt

# 定义每个分段的px和pz函数
def px1(t):
    return np.zeros_like(t)  # 返回 t 对应的零数组

def pz1(t):
    return np.full_like(t, 0.7)  # 返回 pz 为 0.7 的数组

def px2(t):
    return 0.79687 * (t - 0.21)**3 - 0.22761 * (t - 0.21)**2 + 0.02091 * (t - 0.21) + 0.01962

def pz2(t):
    return 11.05639 * (t - 0.21)**3 - 6.56588 * (t - 0.21)**2 + 0.04302 * (t - 0.21) + 0.6992

def px3(t):
    return -2.99073 * (t - 0.61)**3 + 2.83325 * (t - 0.61)**2 + 0.21099 * (t - 0.61) + 0.04366

def pz3(t):
    return -29.49624 * (t - 0.61)**3 + 14.52348 * (t - 0.61)**2 - 0.20425 * (t - 0.61) + 0.37244

def px4(t):
    return -9.69247 * (t - 0.81)**3 + 8.61694 * (t - 0.81)**2 - 2.64865 * (t - 0.81) + 0.12277

def pz4(t):
    return -12.8121 * (t - 0.81)**4 + 11.70907 * (t - 0.81)**3 - 3.46796 * (t - 0.81)**2 + 0.19291 * (t - 0.81) + 0.71137

def px5(t):
    return -3.97864 * (t - 1.21)**3 - 1.0657 * (t - 1.21)**2 + 0.94157 * (t - 1.21) - 0.16949

def pz5(t):
    return 29.29793 * (t - 1.21)**3 - 2.14202 * (t - 1.21)**2 - 2.0858 * (t - 1.21) + 0.64868

def px6(t):
    return 0.12064 * (t - 1.4)**3 - 0.2467 * (t - 1.4)**2 + 0.18683 * (t - 1.4) - 0.05559

def pz6(t):
    return 0.4464 * (t - 1.4)**4 - 0.16473 * (t - 1.4)**3 - 0.89593 * (t - 1.4)**2 + 0.99609 * (t - 1.4) + 0.36358

# 时间区间
t1 = np.linspace(0, 0.2, 100)
t2 = np.linspace(0.21, 0.6, 100)
t3 = np.linspace(0.61, 0.8, 100)
t4 = np.linspace(0.81, 1.2, 100)
t5 = np.linspace(1.21, 1.39, 100)
t6 = np.linspace(1.4, 2.18, 100)

# 计算每个分段的px和pz值
px_values = np.concatenate([px1(t1), px2(t2), px3(t3), px4(t4), px5(t5), px6(t6)])
pz_values = np.concatenate([pz1(t1), pz2(t2), pz3(t3), pz4(t4), pz5(t5), pz6(t6)])
time_values = np.concatenate([t1, t2, t3, t4, t5, t6])

# 绘制px随时间变化的图
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)  # 创建一个2行1列的子图，选择第一个
plt.plot(time_values, px_values, label="px(t)", color='b')
plt.xlabel("time (t)")
plt.ylabel("px")
plt.title("px over time")
plt.grid(True)
plt.legend()

# 绘制pz随时间变化的图
plt.subplot(2, 1, 2)  # 选择第二个子图
plt.plot(time_values, pz_values, label="pz(t)", color='r')
plt.xlabel("time (t)")
plt.ylabel("pz")
plt.title("pz over time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
