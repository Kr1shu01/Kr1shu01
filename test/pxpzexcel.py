import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_px_pz_vs_time(excel_file, sheet_name=0, save_path=None):
    """
    绘制py和pz随时间变化的曲线图。

    参数：
    excel_file (str): Excel文件的路径。
    sheet_name (str or int, optional): 要读取的工作表名称或索引。默认为第一个工作表。
    save_path (str, optional): 图像保存的路径。如果为None，则显示图像。
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(excel_file):
            print(f"文件未找到：{excel_file}")
            return

        # 读取Excel文件
        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # 检查必要的列是否存在（不区分大小写）
        required_columns = ['time', 'px', 'pz']
        df_columns_lower = [col.lower() for col in df.columns]
        if not all(column.lower() in df_columns_lower for column in required_columns):
            print(f"Excel文件中缺少必要的列。需要的列：{required_columns}")
            print(f"当前列：{df.columns.tolist()}")
            return

        # 根据列名不区分大小写提取time、py和pz数据
        column_mapping = {col.lower(): col for col in df.columns}
        time = df[column_mapping['time']]
        px = df[column_mapping['px']]
        pz = df[column_mapping['pz']]

        # 数据预处理：移除缺失值
        df_clean = df.dropna(subset=[column_mapping['time'], column_mapping['px'], column_mapping['pz']])
        time = df_clean[column_mapping['time']]
        px = df_clean[column_mapping['px']]
        pz = df_clean[column_mapping['pz']]

        # 创建子图：2行1列
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # 绘制 py 随时间变化的曲线
        axs[0].plot(time, px, 'b-', label='px', linewidth=2)
        axs[0].set_ylabel('px', fontsize=12)
        axs[0].set_title('px over time', fontsize=14)
        axs[0].legend(loc='upper left')
        axs[0].grid(True)

        # 绘制 pz 随时间变化的曲线
        axs[1].plot(time, pz, 'r-', label='pz', linewidth=2)
        axs[1].set_xlabel('time', fontsize=12)
        axs[1].set_ylabel('pz', fontsize=12)
        axs[1].set_title('pz over time', fontsize=14)
        axs[1].legend(loc='upper left')
        axs[1].grid(True)

        # 调整子图布局
        plt.tight_layout()

        # 保存或显示图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到：{save_path}")
        else:
            plt.show()

    except FileNotFoundError:
        print(f"文件未找到：{excel_file}")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    # 您的Excel文件路径
    excel_file = r"D:\Pyobj\test\pxpz.xlsx"

    # 可选：指定工作表名称或索引（默认为第一个工作表）
    sheet_name = 0  # 或者 'Sheet1' 等

    # 可选：指定保存图像的路径。如果不需要保存，可以将其设置为 None
    save_path = None  # 例如 r"C:\Users\KRISHU\Desktop\py_pz_vs_time.png"

    # 调用函数绘制py和pz随时间变化的曲线图
    plot_px_pz_vs_time(excel_file, sheet_name, save_path)
