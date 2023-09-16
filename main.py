import random
num = random.randint(0,9)
while True:
    x = int(input("输入一个1-9的数字\n"))
    if x > 9 or x < 1 :
        print("Kidding me??")
    elif x > num :
        print("大了哦！再试一次。")
    elif x < num :
        print("小了哦！再试一次。")
    else :
        print("恭喜哦！猜对了呀！答案是"+str(num))
        break