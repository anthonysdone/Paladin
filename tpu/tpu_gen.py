import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import *
from tpu_tasks import *

def gen_tpu(T, program, mem_size=4096):
    sim = Sim()

    mem = sim.reg([0] * mem_size)

    spad_a = sim.reg([0] * (T * T))
    spad_b = sim.reg([0] * (T * T))
    sums_c = sim.reg([0] * (T * T))

    prog = sim.reg(program)

    c_state = sim.reg(TpuState.IF)
    c_pc = sim.reg(0)
    c_op = sim.reg(None)
    c_halt = sim.reg(False)
    c_count = sim.reg(0)

    t_rM = sim.reg(0)
    t_rK = sim.reg(0)
    t_rN = sim.reg(0)
    t_rT = sim.reg(T)
    t_T = sim.reg(T)

    t_m0 = sim.reg(0)
    t_n0 = sim.reg(0)
    t_k0 = sim.reg(0)

    t_rows = sim.reg(0)
    t_cols = sim.reg(0)

    m_abase = sim.reg(0)
    m_bbase = sim.reg(0)
    m_cbase = sim.reg(0)
    m_astep = sim.reg(0)
    m_bstep = sim.reg(0)
    m_cstep = sim.reg(0)

    d_kind = sim.reg(DmaKind.NONE)
    d_base = sim.reg(0)
    d_stride = sim.reg(0)
    d_r = sim.reg(0)
    d_c = sim.reg(0)
    d_done = sim.reg(False)
    d_reads = sim.reg(0)
    d_writes = sim.reg(0)

    s_cycle = sim.reg(0)
    s_count = sim.reg(0)
    s_active = sim.reg(False)

    cycle = sim.reg(0)

    a_feeders = [[sim.reg(0) for _ in range(T)] for _ in range(T)]
    b_feeders = [[sim.reg(0) for _ in range(T)] for _ in range(T)]

    s_sums = [[sim.reg(0) for _ in range(T)] for _ in range(T)]

    a_in_feeders = [sim.reg(0) for _ in range(T)]
    b_in_feeders = [sim.reg(0) for _ in range(T)]

    s_sums_flat = [s_sums[i][j] for i in range(T) for j in range(T)]

    # === DMA ===

    # dma(state, kind, base, step, r, c, done, t_rows, t_cols, mem, spad_a, spad_b, sums_c, T):

    sim.add(dma(
        c_state, 
        d_kind, d_base, d_stride, d_r, d_c, d_done, 
        t_rows, t_cols,
        mem, spad_a, spad_b, sums_c, t_T
    ))

    # === Systolic ===

    sim.add(s_counter(c_state, s_cycle))

    for i in range(t_T.val):
        sim.add(feed_a_row(s_cycle, i, spad_a, a_in_feeders[i], t_T.val))
        sim.add(feed_b_col(s_cycle, i, spad_b, b_in_feeders[i], t_T.val))
    
    def get_a_in(i, j):
        if j == 0:
            return a_in_feeders[i]
        return a_feeders[i][j-1]

    def get_a_out(i, j):
        if j == T - 1:
            return sim.reg(0)
        return a_feeders[i][j]

    def get_b_in(i, j):
        if i == 0:
            return b_in_feeders[j]
        return b_feeders[i-1][j]

    def get_b_out(i, j):
        if i == T - 1:
            return sim.reg(0)
        return b_feeders[i][j]

    for i in range(T):
        for j in range(T):
            sim.add(mac(
                get_a_in(i, j),
                get_b_in(i, j),
                s_sums[i][j],
                get_a_out(i, j),
                get_b_out(i, j),
                s_active
            ))
    
    # === Controller ===

    sim.add(commit(c_state, s_sums_flat, sums_c, t_T))

    sim.add(controller(
        c_state, c_pc, c_op, prog, c_halt, c_count,
        d_kind, d_base, d_stride, d_r, d_c, d_done, d_reads, d_writes,
        s_cycle, s_sums_flat, sums_c, s_count, s_active, 
        t_rM, t_rK, t_rN, t_rT, t_m0, t_n0, t_k0, t_rows, t_cols, t_T,
        m_abase, m_bbase, m_cbase, m_astep, m_bstep, m_cstep
    ))

    sim.add(c_counter(cycle, c_halt))

    outputs = {
        "mem": mem,
        "spad_a": spad_a,
        "spad_b": spad_b,
        "accum_c": sums_c,
        "s_sums": s_sums,
        "state": c_state,
        "pc": c_pc,
        "halted": c_halt,
        "cycle": cycle,
        "instr_count": c_count,
        "dma_reads": d_reads,
        "dma_writes": d_writes,
        "mac_count": s_count,
        "s_cycle": s_cycle,
        "m0": t_m0, "n0": t_n0, "k0": t_k0,
        "rM": t_rM, "rK": t_rK, "rN": t_rN, "rT": t_rT,
    }

    return sim, outputs

def gemm(A, B, C, M, K, N):
    D = [[0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            for k in range(K):
                D[i][j] += A[i][k] * B[k][j]

    for i in range(M):
        for j in range(N):
            D[i][j] += C[i][j]
    return D

def test_tpu(M, K, N, T, A, B, C):
        program = [
            ("mnk", M, N, K),
            ("tile", T), 
            ("gemm",),
            ("halt",)
        ]

        sim, outputs = gen_tpu(T, program)

        mem_init = [0] * 4096
        for r in range(M):
            for c in range(K):
                mem_init[r * K + c] = A[r][c]
        for r in range(K):
            for c in range(N):
                mem_init[M * K + r * N + c] = B[r][c]
        for r in range(M):
            for c in range(N):
                mem_init[M * K + K * N + r * N + c] = C[r][c]

        outputs["mem"].val = mem_init
        outputs["mem"].next = mem_init

        max_cycles = 10000
        cycle_count = 0
        while not outputs["halted"].val and outputs["cycle"].val < max_cycles:
            sim.step()
            cycle_count += 1
        
        actual_c = []
        for r in range(M):
            row = []
            for c in range(N):
                row.append(outputs["mem"].val[M * K + K * N + r * N + c])
            actual_c.append(row)

        print("Actual:")
        for row in actual_c:
            print(row)

        expected = gemm(A, B, C, M, K, N)
        print("Expected:")
        for row in expected:
            print(row)


if __name__ == "__main__":
    M, K, N = 4, 4, 4
    T = 2

    A = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    B = [
        [4, 3, 2, 1],
        [8, 7, 6, 5],
        [12, 11, 10, 9],
        [16, 15, 14, 13]
    ]

    D = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    C = [
        [0, 10, 0, 10], 
        [10, 0, 10, 0], 
        [0, 10, 0, 10], 
        [10, 0, 10, 0]
    ]

    test_tpu(M, K, N, T, A, B, C)

    