import random
import tkinter as tk

class GuessNumberGame:
    def __init__(self, master):
        self.master = master
        master.title("猜数字游戏")

        self.label = tk.Label(master, text="欢迎来到猜数字游戏！请猜一个1到9之间的整数。", font=("Arial", 16))
        self.label.pack(pady=20)

        self.entry = tk.Entry(master)
        self.entry.pack(pady=20)

        self.submit_button = tk.Button(master, text="提交", command=self.guess, font=("Arial", 16))
        self.submit_button.pack(pady=20)

        self.result_label = tk.Label(master, text="", font=("Arial", 16))
        self.result_label.pack(pady=20)

        self.restart_button = tk.Button(master, text="重新开始", command=self.restart, font=("Arial", 16))
        self.restart_button.pack(pady=20)

        self.num = random.randint(1, 9)

    def guess(self):
        try:
            guess = int(self.entry.get())
            if guess > 9 or guess < 1:
                self.result_label.config(text="请输入一个1到9之间的整数！")
            elif guess > self.num:
                self.result_label.config(text="大了哦！再试一次。")
            elif guess < self.num:
                self.result_label.config(text="小了哦！再试一次。")
            else:
                self.result_label.config(text="恭喜哦！猜对了呀！答案是 {}。".format(self.num))
        except ValueError:
            self.result_label.config(text="输入不合法，请输入一个整数！")

    def restart(self):
        self.num = random.randint(1, 9)
        self.result_label.config(text="")
        self.entry.delete(0, tk.END)

root = tk.Tk()
game = GuessNumberGame(root)
root.mainloop()
