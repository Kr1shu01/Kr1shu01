import pandas as pd
import matplotlib.pyplot as plt

# 加载数据，指定文件路径
data = pd.read_excel('C://Users//KRISHU//Desktop//reward-ablation.xlsx', engine='openpyxl')

# 提取数据列
steps = data['Step']
ours = data['ours'].rolling(window=10).mean()  # 应用滚动平均滤波
ours_wo_vae = data['ours-w/o-vae'].rolling(window=10).mean()  # 应用滚动平均滤波
ours_wo_pretraining = data['ours-w/o-pre-traing'].rolling(window=10).mean()  # 应用滚动平均滤波
ours_KL_learning_rate = data['ours-KL-learning-rate'].rolling(window=10).mean()  # 应用滚动平均滤波

# 绘制图形
plt.figure(figsize=(10, 5))
plt.plot(steps, ours, 'r-o', label='ours', markevery=10, markersize=2)  # 调整标记大小
plt.plot(steps, ours_wo_vae, 'y-s', label='ours w/o VAE', markevery=10, markersize=2)  # 调整标记大小
plt.plot(steps, ours_wo_pretraining, 'b-^', label='ours w/o pre-training', markevery=10, markersize=2)  # 调整标记大小
plt.plot(steps, ours_KL_learning_rate, 'g-*', label='ours KL learning rate', markevery=10, markersize=2)  # 调整标记大小

# 添加图例
plt.legend()

# 添加标题和标签
plt.title('Rewards of Different Strategies Over Training Steps')
plt.xlabel('Steps')
plt.ylabel('Rewards')

# 显示图表
plt.show()
