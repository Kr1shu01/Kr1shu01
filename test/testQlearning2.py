import tkinter as tk
import numpy as np
import random
import time

# 设置网格世界的参数
GRID_SIZE = 8  # 网格大小
NUM_ACTIONS = 4  # 可执行的动作数量：上、下、左、右
ACTIONS = ['up', 'down', 'left', 'right']  # 定义动作
START_STATE = (0, 0)  # 起始位置
GOAL_STATE = (GRID_SIZE - 1, GRID_SIZE - 1)  # 目标位置
OBSTACLES = [(7, 3),(0, 6),(7, 4),(7, 5),(5, 6),(6, 5),(6, 1),(1, 2), (1, 1),(3, 0),(4, 0),(3, 3),(4, 3),(5, 3),(5, 2),(1, 3),(1, 4),(1, 5),(2, 5),(1, 6),(3, 7),(2, 2), (3, 2), (4, 5), (5, 5)]  # 定义障碍物的位置

# Q-Learning 设置
LEARNING_RATE = 0.1  # 学习率
DISCOUNT_FACTOR = 0.9  # 折扣因子
EPSILON = 0.1  # 探索概率
Q_TABLE = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))  # 初始化Q表

# 使用Tkinter创建图形界面
class GridWorld(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Grid World with Q-Learning")
        self.geometry(f"{GRID_SIZE * 50}x{GRID_SIZE * 50 + 50}")  # 窗口大小
        self.cells = {}  # 存储单元格对象
        self.create_grid()  # 创建网格
        self.update_state(START_STATE)  # 更新状态显示
        self.create_start_button()  # 创建开始按钮

    def create_grid(self):
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                color = 'white'
                if (x, y) in OBSTACLES:
                    color = 'black'
                elif (x, y) == START_STATE:
                    color = 'green'
                elif (x, y) == GOAL_STATE:
                    color = 'red'
                frame = tk.Frame(self, bg=color, width=50, height=50)
                frame.grid(row=x, column=y)
                self.cells[(x, y)] = frame

    def update_state(self, state):
        # 更新显示状态
        for (x, y), frame in self.cells.items():
            if (x, y) in OBSTACLES:
                frame.config(bg='black')
            elif (x, y) == GOAL_STATE:
                frame.config(bg='red')
            else:
                frame.config(bg='white')
        x, y = state
        if state != GOAL_STATE and state not in OBSTACLES:
            self.cells[(x, y)].config(bg='blue')

    def create_start_button(self):
        # 创建开始训练的按钮
        self.start_button = tk.Button(self, text="Start Training", command=self.start_training)
        self.start_button.grid(row=GRID_SIZE, column=0, columnspan=GRID_SIZE)

    def start_training(self):
        # 开始训练的功能
        self.start_button.config(state="disabled")
        train(self, 1000)  # 指定训练轮数

def get_next_state(state, action):
    # 根据当前状态和动作计算下一个状态
    i, j = state
    if action == 'up' and i > 0 and (i - 1, j) not in OBSTACLES:
        i -= 1
    elif action == 'down' and i < GRID_SIZE - 1 and (i + 1, j) not in OBSTACLES:
        i += 1
    elif action == 'left' and j > 0 and (i, j - 1) not in OBSTACLES:
        j -= 1
    elif action == 'right' and j < GRID_SIZE - 1 and (i, j + 1) not in OBSTACLES:
        j += 1
    return (i, j)

def choose_action(state):
    # 选择一个动作，基于epsilon-greedy策略
    if random.uniform(0, 1) < EPSILON:
        return np.random.choice(ACTIONS)
    else:
        return ACTIONS[np.argmax(Q_TABLE[state[0], state[1], :])]

def update_q_table(state, action, reward, next_state):
    # 更新Q表的值
    i, j = state
    action_idx = ACTIONS.index(action)
    i_next, j_next = next_state
    current_q = Q_TABLE[i, j, action_idx]
    max_future_q = np.max(Q_TABLE[i_next, j_next, :])
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
    Q_TABLE[i, j, action_idx] = new_q

def train(env, episodes=1000, delay=0.01):
    # 训练过程
    for _ in range(episodes):
        state = START_STATE
        while state != GOAL_STATE:
            action = choose_action(state)
            next_state = get_next_state(state, action)
            reward = -1
            if next_state == GOAL_STATE:
                reward = 100
            elif next_state in OBSTACLES:
                reward = -100
            update_q_table(state, action, reward, next_state)
            state = next_state
            env.update_state(state)
            env.update_idletasks()
            env.update()
            time.sleep(delay)  # 暂停一段时间以便观察

if __name__ == "__main__":
    env = GridWorld()
    env.mainloop()
