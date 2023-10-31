import numpy as np
import tkinter as tk
import time

# 初始化参数
gamma = 0.99  # 折扣因子
alpha = 0.1   # 学习率
epsilon = 0.1  # ε-贪婪策略中的ε
delay = 0.5   # 更新界面的延迟时间（秒）

# 环境的状态和动作
states = range(16)
actions = ["up", "down", "left", "right"]
termination_states = [15]

# 奖励函数
rewards = np.full((16,), -1.)
rewards[5] = rewards[7] = rewards[11] = rewards[12] = -10.
rewards[15] = 10.

# 状态转移函数
def step(state, action):
    if state in termination_states:
        return state, 0
    next_state = state
    if action == "up":
        next_state = max(state - 4, 0)
    elif action == "down":
        next_state = state + 4
        if next_state > 15:  # 修复边界问题
            next_state = state
    elif action == "left":
        next_state = state if state % 4 == 0 else state - 1
    elif action == "right":
        next_state = state if (state + 1) % 4 == 0 else state + 1
    return next_state, rewards[next_state]

# Q-表初始化
Q = np.zeros((16, 4))

# Tkinter界面设置
root = tk.Tk()
root.title("Q-Learning Visualization")

canvas = tk.Canvas(root, height=400, width=400)
canvas.pack()

# 画网格
for i in range(1, 4):
    canvas.create_line(i * 100, 0, i * 100, 400)
    canvas.create_line(0, i * 100, 400, i * 100)

# 画网格中的文字
text_mapping = {0: 'S', 5: 'H', 7: 'H', 11: 'H', 12: 'H', 15: 'G'}
for s, t in text_mapping.items():
    x = (s % 4) * 100 + 50
    y = (s // 4) * 100 + 50
    canvas.create_text(x, y, text=t, font=('Arial', 16))

# 画智能体
agent = canvas.create_oval(20, 20, 80, 80, fill="blue")

# 更新智能体的位置
def update_agent_position(state):
    x = (state % 4) * 100 + 50
    y = (state // 4) * 100 + 50
    canvas.coords(agent, x - 30, y - 30, x + 30, y + 30)
    root.update()
    time.sleep(delay)

# Q-Learning训练
def train():
    for episode in range(500):  # 增加训练回合数
        state = 0  # 总是从起点开始
        while state not in termination_states:
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(actions)
            else:
                action = actions[np.argmax(Q[state, :])]
            action_index = actions.index(action)
            next_state, reward = step(state, action)
            print(f"From state {state} to {next_state} by {action}")  # 打印每一步的状态和动作
            next_state_action_values = Q[next_state, :]
            Q[state, action_index] += alpha * (reward + gamma * np.max(next_state_action_values) - Q[state, action_index])
            state = next_state
            update_agent_position(state)

# 启动Tkinter界面
tk.Button(root, text='Start Training', command=train).pack()
root.mainloop()
