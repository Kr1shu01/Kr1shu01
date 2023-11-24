import numpy as np
import tkinter as tk
import time

# 初始化参数
gamma = 0.99  # 折扣因子，用于计算未来奖励的当前价值
alpha = 0.1   # 学习率，用于更新Q值的比例
epsilon = 0.1  # ε-贪婪策略中的ε值，用于平衡探索和利用
delay = 0.05   # 更新界面的延迟时间（秒），用于可视化的流畅性

# 环境的状态和动作
states = range(16)  # 状态集，共16个格子
actions = ["up", "down", "left", "right"]  # 动作集
termination_states = [15]  # 终止状态，即目标状态

# 奖励函数，初始化为-1，特殊状态有不同奖励
rewards = np.full((16,), -1.)
rewards[5] = rewards[7] = rewards[11] = rewards[12] = -10.  # 这些状态为负奖励（Hurt）
rewards[15] = 10.  # 终点的奖励（Victory）

# 状态转移函数
def step(state, action):
    # 如果是终止状态，直接返回当前状态和零奖励
    if state in termination_states:
        return state, 0
    next_state = state
    # 根据动作决定下一个状态
    if action == "up":
        next_state = max(state - 4, 0)  # 向上移动，但不能超出界面
    elif action == "down":
        next_state = state + 4 if state < 12 else state  # 向下移动，但不能超出界面
    elif action == "left":
        next_state = state if state % 4 == 0 else state - 1  # 向左移动，但不能跨越边界
    elif action == "right":
        next_state = state if (state + 1) % 4 == 0 else state + 1  # 向右移动，但不能跨越边界
    # 返回新的状态和对应的奖励
    return next_state, rewards[next_state]

# Q-表初始化，所有的Q值都从0开始
Q = np.zeros((16, 4))

# Tkinter界面设置
root = tk.Tk()
root.title("Q-Learning Visualization")

# 创建一个400x400像素的画布
canvas = tk.Canvas(root, height=400, width=400)
canvas.pack()

# 画网格，4x4的格子，每个格子100x100像素
for i in range(1, 4):
    canvas.create_line(i * 100, 0, i * 100, 400)
    canvas.create_line(0, i * 100, 400, i * 100)

# 画网格中的文字，显示起点、伤害和胜利状态
text_mapping = {0: 'Start', 5: 'Hurt', 6: 'Hurt', 3: 'Hurt', 11: 'Hurt', 12: 'Hurt', 13: 'Hurt', 15: 'Victory'}
for s, t in text_mapping.items():
    x = (s % 4) * 100 + 50
    y = (s // 4) * 100 + 50
    canvas.create_text(x, y, text=t, font=('Arial', 16))

# 画智能体，一个绿色的圆形表示
agent = canvas.create_oval(20, 20, 80, 80, fill="green")

# 更新智能体的位置
def update_agent_position(state):
    x = (state % 4) * 100 + 50  # 计算智能体中心的x坐标
    y = (state // 4) * 100 + 50  # 计算智能体中心的y坐标
    canvas.coords(agent, x - 30, y - 30, x + 30, y + 30)  # 更新智能体的位置
    root.update()  # 更新Tkinter界面
    time.sleep(delay)  # 等待一定的延迟，以便观察

# Q-Learning训练函数
def train():
    for episode in range(300):  # 训练300轮
        state = 0  # 总是从起点开始
        while state not in termination_states:  # 如果没有到达终点则继续训练
            if np.random.uniform(0, 1) < epsilon:  # ε-贪婪策略决定随机探索或选择最好的动作
                action = np.random.choice(actions)
            else:
                action = actions[np.argmax(Q[state, :])]
            action_index = actions.index(action)  # 将动作转换为索引
            next_state, reward = step(state, action)  # 采取动作，获取下一个状态和奖励
            print(f"From state {state} to {next_state} by {action}")  # 打印状态转移信息
            next_state_action_values = Q[next_state, :]  # 获取下一个状态的Q值
            # 更新当前状态的Q值
            Q[state, action_index] += alpha * (reward + gamma * np.max(next_state_action_values) - Q[state, action_index])
            state = next_state  # 进入下一个状态
            update_agent_position(state)  # 更新智能体位置

# 在Tkinter界面添加一个按钮，点击后开始训练
tk.Button(root, text='Start Training', command=train).pack()
root.mainloop()  # 启动Tkinter事件循环
