import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 读取CSV文件
csv_path = "power_trace.csv"
df = pd.read_csv(csv_path)

# 筛选：只保留frame_idx ≥ 40的数据（从索引40开始）
df = df[df["frame_idx"] >= 40].reset_index(drop=True)

# 处理时间戳
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f")

# 创建画布
plt.figure(figsize=(15, 6))

# 1. 按帧索引绘制功耗曲线（从frame=40开始）
plt.subplot(1, 2, 1)
plt.plot(df["frame_idx"], df["gpu_power_w"], color="#1f77b4", linewidth=1.5, label="GPU_Power")
plt.xlabel("Frame", fontsize=12)
plt.ylabel("Power (mW)", fontsize=12)
plt.title("GPU Power vs Frame (Start from Frame 0)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

"""
# 2. 按时间戳绘制功耗曲线（对应frame≥40的时段）
plt.subplot(1, 2, 2)
plt.plot(df["timestamp"], df["gpu_power_w"], color="#ff7f0e", linewidth=1.5, label="GPU_Power")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Power (mW)", fontsize=12)
plt.title("GPU Power vs Time (Start from Frame 0)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45)
"""

avg_power = df["gpu_power_w"].mean()
max_power = df["gpu_power_w"].max()
plt.axhline(y=avg_power, color="red", linestyle="--", label=f"mean: {avg_power:.2f}")
plt.axhline(y=max_power, color="orange", linestyle=":", label=f"peak: {max_power:.2f}")
plt.legend()

# 调整布局并保存
plt.tight_layout()
plt.savefig("power_curve3.png", dpi=300, bbox_inches="tight")
plt.show()