import numpy as np
import matplotlib.pyplot as plt

# 参数
T1 = 80
dh = 5
dh1 = 65

# 时间变量tp，从0到200，步长为1
tp_values = np.arange(0, 2 * T1 + 1, 1)
uff_values = np.zeros((len(tp_values), 12))

# 计算 uff 数组
for i, tp in enumerate(tp_values):
    if 0 < tp <= T1:
        tp0 = tp
        uff_values[i, 3] = 0
        uff_values[i, 9] = 0
        uff_values[i, 5] = (dh + dh1) * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff_values[i, 7] = -2 * dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff_values[i, 11] = (dh - dh1) * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff_values[i, 2] = 0
        uff_values[i, 8] = 0
        uff_values[i, 4] = 0
        uff_values[i, 6] = 0
        uff_values[i, 10] = 0
    elif T1 < tp <= 2 * T1:
        tp0 = tp - T1
        uff_values[i, 3] = 0
        uff_values[i, 9] = 0
        uff_values[i, 5] = 0
        uff_values[i, 7] = 0
        uff_values[i, 11] = 0
        uff_values[i, 2] = 0
        uff_values[i, 8] = 0
        uff_values[i, 4] = (dh + dh1) * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff_values[i, 6] = -2 * dh * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
        uff_values[i, 10] = (dh - dh1) * (-np.cos(np.pi * 2 * tp0 / T1) + 1) / 2
    elif tp >= 2 * T1:
        tp = 0

# 绘制uff数组的不同元素，跳过为0的值
plt.figure(figsize=(10, 6))

for i in range(12):
    # 如果uff[i]中有非零值，则绘制
    if np.any(uff_values[:, i] != 0):
        plt.plot(tp_values, uff_values[:, i], label=f'uff[{i}]')

plt.xlabel('Time (tp)')
plt.ylabel('uff values')
plt.title('uff values over time')
plt.legend()
plt.grid(True)
plt.show()
