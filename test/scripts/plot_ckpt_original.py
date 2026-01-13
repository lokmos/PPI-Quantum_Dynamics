import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
# 请将此处修改为包含 original/ 和 ckpt/ 文件夹的实际路径
BASE_DIR = '/root/zyk/PPI/PPI-Quantum_Dynamics/test/bench/20260109-050450'

# 系统尺寸列表 (L)
L_VALUES = [3, 4, 5, 6, 7, 8]

# 文件名模板 (根据你的文件名格式调整)
# 假设文件名格式为: memlog-dim2-L{L}-d1.00-O4-x0.00-Jz0.10-p0-{mode}.jsonl
FILE_TEMPLATE = "memlog-dim2-L{}-d1.00-O4-x0.00-Jz0.10-p0-{}.jsonl"
# ===========================================

def get_peak_memory(filepath):
    """从 jsonl 日志文件中提取最大 RSS (MB)"""
    max_rss = 0.0
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'rss_mb' in data:
                        rss = data['rss_mb']
                        if rss > max_rss:
                            max_rss = rss
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
        
    return max_rss

def main():
    original_mems = []
    ckpt_mems = []
    valid_Ls = []
    valid_ns = []  # 粒子数

    print(f"Scanning directory: {BASE_DIR}...")

    for L in L_VALUES:
        # 构建文件路径
        path_orig = os.path.join(BASE_DIR, 'original', FILE_TEMPLATE.format(L, 'original'))
        path_ckpt = os.path.join(BASE_DIR, 'ckpt', FILE_TEMPLATE.format(L, 'ckpt'))

        mem_orig = get_peak_memory(path_orig)
        mem_ckpt = get_peak_memory(path_ckpt)

        if mem_orig is not None and mem_ckpt is not None:
            n = L * L  # 对于 2D 系统，粒子数 n = L^2
            valid_Ls.append(L)
            valid_ns.append(n)
            original_mems.append(mem_orig / 1024.0) # 转换为 GB
            ckpt_mems.append(mem_ckpt / 1024.0)     # 转换为 GB
            print(f"L={L}, n={n}: Original={mem_orig/1024:.2f} GB, Ckpt={mem_ckpt/1024:.2f} GB")

    if not valid_Ls:
        print("No valid data found. Please check the directory path and file names.")
        return

    # 计算内存节省统计
    print("\n" + "="*60)
    print("Memory Reduction Statistics:")
    print("="*60)
    for i, (L, n) in enumerate(zip(valid_Ls, valid_ns)):
        reduction = (original_mems[i] - ckpt_mems[i]) / original_mems[i] * 100
        print(f"n={n:2d} (L={L}): {original_mems[i]:6.2f} GB → {ckpt_mems[i]:6.2f} GB | Reduction: {reduction:5.1f}%")
    avg_reduction = np.mean([(original_mems[i] - ckpt_mems[i]) / original_mems[i] * 100 for i in range(len(valid_Ls))])
    print("="*60)
    print(f"Average Memory Reduction: {avg_reduction:.1f}%")
    print("="*60 + "\n")

    # ================= 绘图 =================
    plt.figure(figsize=(10, 7))
    
    # 设置柱状图宽度
    bar_width = 0.35
    index = np.arange(len(valid_ns))

    # 绘制柱状图
    bars1 = plt.bar(index, original_mems, bar_width, 
                    label='Standard Forward Mode', 
                    color='#d62728', alpha=0.85, edgecolor='black', linewidth=0.8)
    bars2 = plt.bar(index + bar_width, ckpt_mems, bar_width, 
                    label='Checkpointing Mode', 
                    color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.8)

    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    add_labels(bars1)
    add_labels(bars2)

    # 图表装饰
    plt.xlabel('Number of Sites ($n$)', fontsize=16, fontweight='bold')
    plt.ylabel('Peak Memory Usage (GB)', fontsize=16, fontweight='bold')
    plt.title('Memory Scaling: Flow Equations with Checkpointing', fontsize=17, fontweight='bold', pad=20)
    plt.xticks(index + bar_width / 2, [f'{n}' for n in valid_ns], fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13, loc='upper left', frameon=True, shadow=True)
    plt.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)

    # 添加系统信息注释
    info_text = f"2D Spinless Fermion ($\\mathrm{{dim}}=2$)\nSystem sizes: $L \\in [{min(valid_Ls)}, {max(valid_Ls)}]$"
    plt.text(0.98, 0.97, info_text, 
             transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, edgecolor='gray'))

    plt.tight_layout()
    
    # 保存图片
    output_file = 'memory_scaling_checkpointing.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    print(f"Figure ready for publication.")
    plt.show()

if __name__ == "__main__":
    main()