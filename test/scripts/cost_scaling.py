import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import os

# ==========================================
# 1. 核心瓶颈：全连接混沌系统 (N=10)
# ==========================================
def build_system(N=10):
    dim = 2 ** N
    print(f"--- Building System (N={N}, Matrix Dim={dim}x{dim}) ---")
    
    # 构建基础算符
    sz = sparse.csr_matrix([[0.5, 0], [0, -0.5]])
    sx = sparse.csr_matrix([[0, 0.5], [0.5, 0]])
    id_op = sparse.eye(2)
    
    def op_at(op, i):
        ops = [id_op] * N
        ops[i] = op
        res = ops[0]
        for k in range(1, N):
            res = sparse.kron(res, ops[k])
        return res

    # 构建哈密顿量 (全连接 SYK-like 模型)
    # 这种模型能确保矩阵是全稠密的，没有任何稀疏优化空间
    H = sparse.csr_matrix((dim, dim))
    np.random.seed(42)
    # All-to-all interaction
    for i in range(N):
        for j in range(i + 1, N):
            J = np.random.randn()
            H += J * (op_at(sz, i) @ op_at(sz, j))
            H += J * (op_at(sx, i) @ op_at(sx, j))
    
    return H.toarray().astype(np.complex128)

# ==========================================
# 2. 纯粹的算力测试函数
# ==========================================
def comm(A, B):
    return A @ B - B @ A

def benchmark_rhs_cost(H, num_ops, trials=20):
    """
    不运行 solve_ivp，直接测量 dO/dt = i[H, O] 的函数调用耗时。
    这是最纯粹的"计算代价"指标，排除了求解器步长控制的干扰。
    """
    dim = H.shape[0]
    
    # 随机生成 num_ops 个算符
    ops = [np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim) for _ in range(num_ops)]
    
    # 预热 (Warmup)
    for O in ops:
        _ = comm(H, O)
        
    # 正式计时
    t0 = time.time()
    for _ in range(trials):
        # 模拟一次完整的 RHS 计算：对所有算符求导
        # 这里模拟的是求解器在一步(Step)内必须完成的工作量
        for O in ops:
            _ = comm(H, O)
            
    total_time = time.time() - t0
    avg_time_per_call = total_time / trials
    return avg_time_per_call

# ==========================================
# 3. 运行对比实验
# ==========================================
def run_scaling_proof():
    # 准备系统
    # N=10 (1024x1024) 是能够明显看出矩阵乘法开销的底线
    H = build_system(N=10) 
    
    # 测试用例：算符数量
    # 我们测试到 20 个算符，足以画出一条惊人的斜线
    op_counts = [1, 2, 5, 10, 15, 20]
    
    times = []
    print("\n--- Starting Cost Scaling Benchmark ---")
    print(f"{'Num Ops':<10} | {'Cost per Step (s)':<20} | {'Scaling Factor':<15}")
    print("-" * 50)
    
    base_cost = 0
    
    for k in op_counts:
        cost = benchmark_rhs_cost(H, num_ops=k, trials=10)
            
        times.append(cost)
        
        # 计算倍率
        if k == 1: base_cost = cost
        ratio = cost / base_cost if base_cost > 0 else 0
        
        print(f"{k:<10} | {cost:.5f} s            | {ratio:.1f}x")

    # ==========================================
    # 4. 绘图：这才是你要的"明显差距"
    # ==========================================
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # 绘制实际数据点
    plt.plot(op_counts, times, 'ro-', markersize=8, linewidth=2, label='Measured Cost per Step')
    
    # 绘制理想的线性拟合，展示计算复杂度的不可避免性
    if len(op_counts) > 2:
        z = np.polyfit(op_counts, times, 1)
        p = np.poly1d(z)
        plt.plot(op_counts, p(op_counts), 'b--', alpha=0.5, label=f'Linear Fit (Slope={z[0]:.4f} s/op)')
    
    plt.title(f"Computational Reality: The Cost of Forward Evolution\n(Dim=1024 Full Dense Matrix)", fontsize=14)
    plt.xlabel("Number of Operators to Track", fontsize=12)
    plt.ylabel("Wall-Clock Time per Integration Step (seconds)", fontsize=12)
    # 强制横轴刻度为整数（并优先显示我们采样的整数点）
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(op_counts)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # 添加关键标注
    plt.text(op_counts[-1], times[-1], f"{times[-1]:.3f}s", ha='right', va='bottom', fontsize=12, fontweight='bold', color='red')
    plt.text(op_counts[0], times[0], f"{times[0]:.3f}s", ha='left', va='top', fontsize=12, fontweight='bold', color='red')

    save_path = os.path.join(os.getcwd(), 'cost_scaling_gap.png')
    plt.savefig(save_path)
    print(f"\n[Image Saved]: {save_path}")

if __name__ == "__main__":
    run_scaling_proof()