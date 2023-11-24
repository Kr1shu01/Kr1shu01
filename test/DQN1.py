import tkinter as tk
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from time import sleep

# 环境定义
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.end = (size - 1, size - 1)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:   # 上
            x = max(0, x - 1)
        elif action == 1: # 下
            x = min(self.size - 1, x + 1)
        elif action == 2: # 左
            y = max(0, y - 1)
        elif action == 3: # 右
            y = min(self.size - 1, y + 1)

        self.state = (x, y)

        if self.state == self.end:
            return self.state, 1, True  # 到达终点
        else:
            return self.state, 0, False

# DQN模型定义
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 经验回放缓冲区定义
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# 可视化环境类定义
class GridWorldGUI:
    def __init__(self, env, model, target_model, optimizer, buffer, size=400):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.buffer = buffer
        self.size = size
        self.cell_size = size // env.size
        self.window = tk.Tk()
        self.window.title("Grid World")
        self.canvas = tk.Canvas(self.window, width=size, height=size)
        self.canvas.pack()
        self.draw_grid()
        self.agent = self.canvas.create_rectangle(0, 0, self.cell_size, self.cell_size, fill="blue")

        # 控制按钮
        self.train_button = tk.Button(self.window, text="Train", command=self.start_training)
        self.train_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(self.window, text="Stop", command=self.stop_training)
        self.stop_button.pack(side=tk.LEFT)

        self.running = False

    def draw_grid(self):
        # 绘制网格
        for i in range(self.env.size):
            for j in range(self.env.size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")
        x_end, y_end = self.env.end
        self.canvas.create_rectangle(y_end * self.cell_size, x_end * self.cell_size,
                                     (y_end + 1) * self.cell_size, (x_end + 1) * self.cell_size,
                                     fill="green")

    def update_agent(self, state):
        # 更新代理位置
        x, y = state
        self.canvas.coords(self.agent, y * self.cell_size, x * self.cell_size,
                           (y + 1) * self.cell_size, (x + 1) * self.cell_size)
        self.window.update()

    def start_training(self):
        # 开始训练
        self.running = True
        self.train_dqn()

    def stop_training(self):
        # 停止训练
        self.running = False

    def train_dqn(self):
        # DQN训练过程
        episodes = 1000
        gamma = 0.9
        epsilon = 0.8
        batch_size = 64

        for episode in range(episodes):
            state = self.env.reset()
            self.update_agent(state)
            done = False
            while not done and self.running:
                # 选择动作
                state_tensor = torch.FloatTensor(state)
                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                        action = torch.argmax(q_values).item()

                # 执行动作
                next_state, reward, done = self.env.step(action)
                self.update_agent(next_state)
                sleep(0.01)  # 添加延时以便观察

                # 存储经验
                self.buffer.push(state, action, reward, next_state, done)

                # 从回放缓冲区中采样经验
                if len(self.buffer) > batch_size:
                    batch = self.buffer.sample(batch_size)
                    # ...（后续训练步骤与之前相同）...

                state = next_state

                # 更新目标网络
                if episode % 10 == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

# 超参数设置
learning_rate = 0.01
gamma = 0.9
epsilon = 0.8
episodes = 1000
hidden_size = 64
action_size = 4  # 上下左右
state_size = 2   # 网格的x和y坐标
buffer_size = 10000
batch_size = 64

# 初始化环境
env = GridWorld()

# 初始化网络和优化器
model = DQN(state_size, hidden_size, action_size)
target_model = DQN(state_size, hidden_size, action_size)  # 目标网络
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
replay_buffer = ReplayBuffer(buffer_size)

# 其余代码保持不变

# 主窗口事件循环
gui = GridWorldGUI(env, model, target_model, optimizer, replay_buffer)
gui.window.mainloop()