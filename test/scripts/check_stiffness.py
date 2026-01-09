import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import os

# ==========================================
# 1. 构建全连接混沌系统 (All-to-All Quantum Chaos)
# ==========================================
def build_chaos_system(N=10):
    dim = 2 ** N
    print(f"--- SYSTEM: All-to-All Chaos Model (N={N}, Dim={dim}x{dim}) ---")
    print(f"Topology: Fully Connected (Infinite coordination number)")
    
    # 基础 Pauli 矩阵
    sz = sparse.csr_matrix([[0.5, 0], [0, -0.5]])
    sx = sparse.csr_matrix([[0, 0.5], [0.5, 0]])
    sy = sparse.csr_matrix([[0, -0.5j], [0.5j, 0]])
    id_op = sparse.eye(2)
    
    # 辅助函数：生成单点算符
    def get_op_at(op, i):
        ops = [id_op] * N
        ops[i] = op
        res = ops[0]
        for k in range(1, N):
            res = sparse.kron(res, ops[k])
        return res

    # 1. 构建哈密顿量 (全连接自旋玻璃)
    # H = sum_{i<j} J_{ij} (X_i X_j + Y_i Y_j + Z_i Z_j) + sum h_i Z_i
    # 这种模型没有空间概念，复杂度传播极快
    print("Building Dense Hamiltonian (High Complexity)...")
    H = sparse.csr_matrix((dim, dim))
    
    np.random.seed(666) # 固定种子，保证混沌确定性
    
    # 随机耦合强度 J_ij
    couplings = 0
    for i in range(N):
        for j in range(i + 1, N):
            J = np.random.normal(0, 1.0) # 强无序
            H += J * (get_op_at(sx, i) @ get_op_at(sx, j))
            H += J * (get_op_at(sy, i) @ get_op_at(sy, j))
            H += J * (get_op_at(sz, i) @ get_op_at(sz, j))
            couplings += 1
            
    # 随机纵场
    for i in range(N):
        h = np.random.normal(0, 0.5)
        H += h * get_op_at(sz, i)

    print(f"Hamiltonian Constructed with {couplings} interaction terms.")
    H_dense = H.toarray().astype(np.complex128)
    
    # 2. 生成 10 个算符 (不同位置的 Sz, Sx 混合)
    print("Generating 10 Operators...")
    ops_list = []
    for i in range(N):
        # 混合一下，前一半是 Z，后一半是 X，增加算符多样性
        op_type = sz if i < N//2 else sx
        ops_list.append(get_op_at(op_type, i).toarray().astype(np.complex128))
        
    return H_dense, ops_list

# ==========================================
# 2. 演化核心 (Matrix Multiplication Bottleneck)
# ==========================================
def comm(A, B):
    return A @ B - B @ A

def chaos_evolution(t, y_flat, dim, num_ops):
    # 解析
    H = y_flat[:dim*dim].reshape(dim, dim)
    
    # H 是时间平移不变的 -> dH = 0
    # 我们依然保持 0 计算量给 Group A
    dH = np.zeros_like(H) 
    derivs = [dH.ravel()]
    
    # 算符演化
    # 每一个算符都要经历全连接 H 的洗礼
    # 这里的 H 是完全稠密的，没有任何稀疏结构可利用
    start_idx = dim*dim
    for _ in range(num_ops):
        end_idx = start_idx + dim*dim
        O_curr = y_flat[start_idx:end_idx].reshape(dim, dim)
        
        # 核心负载：1024x1024 复数矩阵乘法
        dO = 1j * comm(H, O_curr)
        derivs.append(dO.ravel())
        
        start_idx = end_idx
        
    return np.concatenate(derivs)

# ==========================================
# 3. 运行对比
# ==========================================
def run_chaos_benchmark():
    N = 10
    H_init, all_ops = build_chaos_system(N)
    dim = H_init.shape[0]
    
    # 演化参数
    t_span = (0, 0.5) # 全连接模型演化极快，0.5 已经很长了
    rtol_val = 1e-9   # 高精度
    atol_val = 1e-11
    
    # 实验分组
    cases = [0, 1, 5, 10]
    colors = ['blue', 'green', 'orange', 'red']
    results = {}
    
    print(f"\n--- Starting Chaos Benchmark (t={t_span[1]}) ---")
    
    for k in cases:
        print(f"\n>> Case: H + {k} Operators ...")
        
        # 准备数据
        current_ops = all_ops[:k]
        y0 = np.concatenate([H_init.ravel()] + [op.ravel() for op in current_ops])
        
        t0 = time.time()
        sol = solve_ivp(chaos_evolution, t_span, y0, args=(dim, k),
                        method='RK45', rtol=rtol_val, atol=atol_val)
        elapsed = time.time() - t0
        
        steps = len(sol.t)
        print(f"   [DONE] Time: {elapsed:.3f}s | Steps: {steps}")
        results[k] = {'time': elapsed, 'steps': steps, 't': sol.t, 'dt': np.diff(sol.t)}

    # ==========================================
    # 4. 绘图（只画第一张图）
    # ==========================================
    plt.figure(figsize=(12, 6))

    # --- 图: 步长坍塌 (dt vs t) ---
    for i, k in enumerate(cases):
        t_vals = results[k]['t'][:-1]
        dt_vals = results[k]['dt']
        label = f"H + {k} Ops"
        lw = 2.5 if k == 0 else 1.0
        plt.semilogy(t_vals, dt_vals, color=colors[i], lw=lw, label=label)

    plt.title(f"Complexity Explosion: Step Size in Chaos Model (N={N})", fontsize=14)
    plt.ylabel("Solver Step Size dt (Log)", fontsize=12)
    plt.xlabel("Evolution Time", fontsize=12)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(os.getcwd(), 'chaos_stiffness_proof.png')
    plt.savefig(save_path)
    print(f"\n[Proof Saved]: {save_path}")

if __name__ == "__main__":
    run_chaos_benchmark()