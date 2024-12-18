import numpy as np
import matplotlib.pyplot as plt

# 定义参数
T1 = 80
dh = 45
d0 = 0
time = np.linspace(0, 2*T1, 1000)  # 时间范围从0到2*T1

# 初始化uff数组
uff = np.zeros((12, len(time)))

# 计算uff数组每个元素随时间的变化
for i, tp in enumerate(time):
    tp0 = tp  # 时间点
    if tp >= 0 and tp <= T1:  # 时间段 0 到 T1
        uff[5][i] = d0 + dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff[4][i] = d0
        uff[7][i] = -2 * d0 - 2 * dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff[6][i] = -2 * d0
        uff[11][i] = d0 - dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff[10][i] = d0
        uff[3][i] = 0
        uff[9][i] = 0
        uff[2][i] = 0
        uff[8][i] = 0
    elif tp > T1 and tp <= 2 * T1:  # 时间段 T1 到 2*T1
        uff[4][i] = d0 + dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff[5][i] = d0
        uff[6][i] = -2 * d0 - 2 * dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff[7][i] = -2 * d0
        uff[10][i] = d0 - dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff[11][i] = d0
        uff[3][i] = 0
        uff[9][i] = 0
        uff[2][i] = 0
        uff[8][i] = 0
    else:  # 重置tp
        tp = 0

# 绘制所有uff元素的图形
plt.figure(figsize=(10, 8))

# 绘制每个uff[0]到uff[11]随时间的变化曲线
for i in range(12):
    plt.plot(time, uff[i], label=f'uff[{i}]')

plt.xlabel('Time (tp)')
plt.ylabel('Joint Position (uff)')
plt.legend()
plt.title('Joint Positions Over Time (Walk)')
plt.grid(True)
plt.show()
