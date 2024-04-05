import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
import time

# 环境设置
grid_size = 5
trap_positions = [(1, 1), (2, 3), (3, 3)]
start_position = (0, 0)
end_position = (4, 4)
cell_size = 80
# 状态编码
def state_to_tensor(state):
    grid = np.zeros((grid_size, grid_size))
    grid[state] = 1
    return torch.tensor(grid.flatten(), dtype=torch.float32)

# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(grid_size * grid_size, 4)  # 4个动作

    def forward(self, state):
        return torch.softmax(self.fc(state), dim=-1)

policy_net = PolicyNetwork()
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)



def step(state, action):
    y, x = state
    if action == 0 and y > 0:   # 上
        y -= 1
    elif action == 1 and y < grid_size - 1: # 下
        y += 1
    elif action == 2 and x > 0: # 左
        x -= 1
    elif action == 3 and x < grid_size - 1: # 右
        x += 1

    new_state = (y, x)
    reward = -1 if new_state in trap_positions else 0
    done = new_state == end_position
    return new_state, reward, done


def train_and_visualize(policy_net, optimizer, episodes, gamma, visualize_interval, canvas, agent, window):
    def visualize_current_policy():
        def step_agent(state):
            if state == end_position:
                return
            state_tensor = state_to_tensor(state)
            action_probs = policy_net(state_tensor).detach().numpy()
            action = np.argmax(action_probs)
            new_state, _, _ = step(state, action)
            canvas.moveto(agent, new_state[1]*cell_size, new_state[0]*cell_size)
            window.update()
            window.after(500, lambda: step_agent(new_state))

        step_agent(start_position)

    for episode in range(episodes):
        state = start_position
        rewards = []
        log_probs = []
        done = False

        while not done:
            state_tensor = state_to_tensor(state)
            action_probs = policy_net(state_tensor)
            action = np.random.choice(4, p=action_probs.detach().numpy())
            log_prob = torch.log(action_probs[action])

            new_state, reward, done = step(state, action)

            log_probs.append(log_prob)
            rewards.append(reward)
            state = new_state

        # 累积折扣奖励
        discounted_rewards = [gamma ** i * r for i, r in enumerate(rewards)]
        policy_loss = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            # 将标量转换为一维张量
            loss = -log_prob * Gt
            policy_loss.append(loss.unsqueeze(0))

        # 现在可以安全地使用 torch.cat
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        # 间歇性可视化
        if episode % visualize_interval == 0:
            visualize_current_policy()
# 初始化 canvas 和 agent
window = tk.Tk()
canvas = tk.Canvas(window, width=grid_size*cell_size, height=grid_size*cell_size)
canvas.pack()
# ... 创建网格和陷阱 ...
agent = canvas.create_oval(0, 0, cell_size, cell_size, fill="yellow")

# 启动训练和可视化
train_and_visualize(policy_net, optimizer, 1000, 0.99, 100, canvas, agent, window)
window.mainloop()

# 训练策略网络并间歇性可视化
train_and_visualize(policy_net, optimizer, episodes=1000, gamma=0.99, visualize_interval=100)

# 控制按钮和窗口主循环（与之前相同）
# Tkinter界面设置
# 初始化 canvas 和 agent
window = tk.Tk()
canvas = tk.Canvas(window, width=grid_size*cell_size, height=grid_size*cell_size)
canvas.pack()

# 绘制网格和陷阱
for i in range(grid_size):
    for j in range(grid_size):
        color = "white"
        if (i, j) in trap_positions:
            color = "red"
        elif (i, j) == start_position:
            color = "green"
        elif (i, j) == end_position:
            color = "blue"
        canvas.create_rectangle(j*cell_size, i*cell_size, (j+1)*cell_size, (i+1)*cell_size, fill=color)

# 创建智能体
agent = canvas.create_oval(0, 0, cell_size, cell_size, fill="yellow")

def move_agent():
    def step_agent(state):
        if state == end_position:
            return
        state_tensor = state_to_tensor(state)
        action_probs = policy_net(state_tensor).detach().numpy()
        action = np.argmax(action_probs)
        new_state, _, _ = step(state, action)
        canvas.moveto(agent, new_state[1]*cell_size, new_state[0]*cell_size)
        window.update()
        window.after(500, lambda: step_agent(new_state))  # 使用lambda传递新状态

    step_agent(start_position)
# 控制按钮
start_button = tk.Button(window, text="Start", command=move_agent)
start_button.pack()

window.mainloop()