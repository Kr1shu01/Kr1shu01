import numpy as np
import matplotlib.pyplot as plt

# 参数
T1 = 80
dh = 60  # 假设dh的值为60
d0 = 0  # d0设为0

# 时间变量tp，从0到2*T1，步长为1
tp_values = np.arange(0, 2 * T1 + 1, 1)
uff_values = np.zeros((len(tp_values), 12))

# 计算 uff 数组
for i, tp in enumerate(tp_values):
    # 第一个周期 0 <= tp <= T1
    if 0 <= tp <= T1:
        tp0 = tp
        uff_values[i, 5] = d0 + dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff_values[i, 4] = d0
        uff_values[i, 7] = -2 * d0 - 2 * dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff_values[i, 6] = -2 * d0
        uff_values[i, 11] = d0 - dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff_values[i, 10] = d0
        uff_values[i, 3] = 0
        uff_values[i, 9] = 0
        uff_values[i, 2] = 0
        uff_values[i, 8] = 0
    # 第二个周期 T1 < tp < 2*T1
    elif T1 < tp <= 2 * T1:
        tp0 = tp
        uff_values[i, 5] = d0 + dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff_values[i, 4] = d0
        uff_values[i, 7] = -2 * d0 - 2 * dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff_values[i, 6] = -2 * d0
        uff_values[i, 11] = d0 - dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff_values[i, 10] = d0
        uff_values[i, 3] = 0
        uff_values[i, 9] = 0
        uff_values[i, 2] = 0
        uff_values[i, 8] = 0
    # 重置 tp
    elif tp > 2 * T1:
        tp = 0

# 绘制uff数组的不同元素，跳过为0的值
plt.figure(figsize=(10, 6))

for i in range(12):
    # 如果uff[i]中有非零值，则绘制
    if np.any(uff_values[:, i] != 0):
        plt.plot(tp_values, uff_values[:, i], label=f'uff[{i}]')

plt.xlabel('Time (tp)')
plt.ylabel('uff values')
plt.title('uff values over time (with walk state)')
plt.legend()
plt.grid(True)
plt.show()
