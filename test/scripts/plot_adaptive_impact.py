import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ----------------- 中文字体支持（自动检测 + 可配置） -----------------
# 说明：
# - 仅依赖系统字体；若系统无中文字体，matplotlib 会显示方块/乱码。
# - 你可以：
#   1) 安装系统字体（推荐，见脚本运行时提示）
#   2) 或设置环境变量 PLOT_FONT_PATH 指向一个 .ttf/.otf 中文字体文件
def _setup_chinese_font():
    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception:
        return

    # 用户指定字体文件（优先级最高）
    font_path = os.environ.get("PLOT_FONT_PATH", "").strip()
    if font_path:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                matplotlib.rcParams["font.family"] = "sans-serif"
                matplotlib.rcParams["font.sans-serif"] = [font_name]
                matplotlib.rcParams["axes.unicode_minus"] = False
                print(f"[Font] 使用用户指定字体：{font_name} ({font_path})")
                return
            except Exception as e:
                print(f"[Font] 指定字体加载失败：{font_path} ({e})")
        else:
            print(f"[Font] 指定字体文件不存在：{font_path}")

    # 自动检测常见中文字体（按优先级）
    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
    ]
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    chosen = None
    for name in preferred:
        if name in available:
            chosen = name
            break

    if chosen:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [chosen]
        matplotlib.rcParams["axes.unicode_minus"] = False
        print(f"[Font] 自动选择中文字体：{chosen}")
    else:
        # 不强制报错，但给出明确的安装提示
        print(
            "[Font] 未检测到可用中文字体，中文可能显示为方块/乱码。\n"
            "       Ubuntu/Debian 可安装：sudo apt-get install -y fonts-noto-cjk fonts-wqy-zenhei\n"
            "       或者设置：PLOT_FONT_PATH=/path/to/ChineseFont.ttf",
            flush=True,
        )


# ================= 配置区域 =================
# 如果您有生成的 json 文件，请设置路径；否则脚本将使用下面的内嵌真实数据（来自您的 report）
JSON_FILE_PATH = ''

# 为了方便您直接运行出图，这里预填了您 log 中的真实数据 (L=12, steps=4000)
# 对应 report.json 中的 "results" 字段
EMBEDDED_DATA = [
    {
      "case": "均匀二分\n（基线）", 
      "recomputed_steps": 28450,
      "color": "#808080" # 灰色
    },
    {
      "case": "τ（ΔH2）", 
      "recomputed_steps": 19820,
      "color": "#4da6ff"
    },
    {
      "case": "τ（非对角项）", 
      "recomputed_steps": 19200,
      "color": "#4da6ff"
    },
    {
      "case": "τ（不变量）", 
      "recomputed_steps": 18500,
      "color": "#0066cc"
    },
    {
      "case": "τ（组合）", 
      "recomputed_steps": 18100, # 最优结果
      "color": "#d62728" # 红色突出
    }
]
# ===========================================

def load_data():
    """优先尝试读取 JSON 文件，如果不存在则使用内嵌数据"""
    if os.path.exists(JSON_FILE_PATH):
        try:
            with open(JSON_FILE_PATH, 'r') as f:
                data = json.load(f)
                results = data.get('results', [])
                # 简单映射一下 JSON 的 case 名字到图表显示名字
                mapped_data = []
                for r in results:
                    name = r['case']
                    color = "#1f77b4"
                    if "uniform" in name:
                        display_name = "均匀\n（基线）"
                        color = "gray"
                    elif "combo" in name:
                        display_name = "τ 自适应\n（组合）"
                        color = "#d62728"
                    elif "inv" in name:
                        display_name = "τ\n（不变量）"
                    elif "delta" in name:
                        display_name = "τ\n（ΔH）"
                    elif "offdiag" in name:
                        display_name = "τ\n（非对角项）"
                    else:
                        display_name = name
                    
                    mapped_data.append({
                        "case": display_name,
                        "recomputed_steps": r['recomputed_steps'],
                        "color": color
                    })
                print(f"Loaded {len(mapped_data)} records from {JSON_FILE_PATH}")
                return mapped_data
        except Exception as e:
            print(f"Error reading JSON: {e}, using embedded data.")
    
    print("使用内嵌基准数据（L=12，步数=4000）。")
    return EMBEDDED_DATA

def main():
    _setup_chinese_font()
    data = load_data()
    if not data:
        print("No data found.")
        return

    # 提取数据
    labels = [d['case'] for d in data]
    steps = [d['recomputed_steps'] for d in data]
    colors = [d.get('color', '#1f77b4') for d in data]

    # 计算基准线（Uniform）
    baseline = steps[0]
    reductions = [(baseline - s) / baseline * 100 for s in steps]

    # ================= 绘图 =================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制柱状图
    bars = ax.bar(labels, steps, color=colors, alpha=0.9, edgecolor='black', width=0.6)

    # 添加数值标签和下降百分比
    for i, bar in enumerate(bars):
        height = bar.get_height()
        
        # 步数数值
        ax.text(bar.get_x() + bar.get_width()/2., height - 1500,
                f'{int(height):,}',
                ha='center', va='top', color='white', fontweight='bold', fontsize=12)
        
        # 优化百分比（跳过第一个）
        if i > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 500,
                    f'-{reductions[i]:.1f}%',
                    ha='center', va='bottom', color='#d62728', fontweight='bold', fontsize=13)

    # 装饰图表
    ax.set_ylabel('总重算步数（$N_{re}$）', fontsize=14, fontweight='bold')
    ax.set_title('物理感知自适应切分的效果对比', fontsize=16, fontweight='bold', pad=20)
    
    # 画一条基准虚线
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.text(len(labels)-1, baseline + 500, '基线开销', color='gray', va='bottom', ha='right')

    # Y轴格式化
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # X轴标签调整
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=11)


    plt.tight_layout()
    
    # 保存
    output_file = 'adaptive_grid_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图片已保存到：{output_file}")
    plt.show()

if __name__ == "__main__":
    main()