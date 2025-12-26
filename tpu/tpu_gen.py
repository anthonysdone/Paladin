import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import *
from tpu_tasks import *

def gen_tpu(N, program, mem_size=4096):
    sim = Sim()

    mem = sim.reg([0] * mem_size)

    spad_a = sim.reg([0] * (N * N))
    spad_b = sim.reg([0] * (N * N))
    spad_c = sim.reg([0] * (N * N))

    prog = sim.reg(program)

    c_state = sim.reg(TpuState.IF)
    c_pc = sim.reg(0)
    c_op = sim.reg(None)
    c_halt = sim.reg(False)
    c_count = sim.reg(0)

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

    a_feeders = [[sim.reg(0) for _ in range(N)] for _ in range(N)]
    b_feeders = [[sim.reg(0) for _ in range(N)] for _ in range(N)]

    s_sums = [[sim.reg(0) for _ in range(N)] for _ in range(N)]

    a_in_feeders = [sim.reg(0) for _ in range(N)]
    b_in_feeders = [sim.reg(0) for _ in range(N)]

    s_sums_flat = [s_sums[i][j] for i in range(N) for j in range(N)]

    # === DMA ===

    sim.add(dma(
        c_state, 
        d_kind, d_base, d_stride, d_r, d_c, d_done, 
        mem, spad_a, spad_b, spad_c, N
    ))

    # === Systolic ===

    sim.add(s_counter(c_state, s_cycle))

    for i in range(N):
        sim.add(feed_a_row(s_cycle, i, spad_a, a_in_feeders[i], N))
        sim.add(feed_b_col(s_cycle, i, spad_b, b_in_feeders[i], N))
    
    def get_a_in(i, j):
        if j == 0:
            return a_in_feeders[i]
        return a_feeders[i][j-1]

    def get_a_out(i, j):
        if j == N - 1:
            return sim.reg(0)
        return a_feeders[i][j]

    def get_b_in(i, j):
        if i == 0:
            return b_in_feeders[j]
        return b_feeders[i-1][j]

    def get_b_out(i, j):
        if i == N - 1:
            return sim.reg(0)
        return b_feeders[i][j]

    for i in range(N):
        for j in range(N):
            sim.add(mac(
                get_a_in(i, j),
                get_b_in(i, j),
                s_sums[i][j],
                get_a_out(i, j),
                get_b_out(i, j),
                s_active
            ))
    
    # === Controller ===

    sim.add(controller(
        c_state, c_pc, c_op, prog, c_halt, c_count,
        d_kind, d_base, d_stride, d_r, d_c, d_done, d_reads, d_writes,
        s_cycle, s_sums_flat, spad_c, s_count, s_active, N
    ))

    sim.add(c_counter(cycle, c_halt))

    outputs = {
        "mem": mem,
        "spad_a": spad_a,
        "spad_b": spad_b,
        "spad_c": spad_c,
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
    }

    return sim, outputs

def gemm(A, B, C, N):
    D = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                D[i][j] += A[i][k] * B[k][j]

    for i in range(N):
        for j in range(N):
            D[i][j] += C[i][j]
    return D


if __name__ == "__main__":
    N = 4

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

    C = [
        [0, 10, 0, 10], 
        [10, 0, 10, 0], 
        [0, 10, 0, 10], 
        [10, 0, 10, 0]
    ]

    program = [
        ("lda", 0, N),
        ("ldb", N * N, N), 
        ("ldc", 2 * N * N, N),
        ("gemm",),
        ("stc", 2 * N * N, N),
        ("halt",)
    ]

    sim, outputs = gen_tpu(N, program)

    mem_init = [0] * 4096
    for r in range(N):
        for c in range(N):
            mem_init[r * N + c] = A[r][c]
            mem_init[N * N + r * N + c] = B[r][c]
            mem_init[2 * N * N + r * N + c] = C[r][c]
    outputs["mem"].val = mem_init
    outputs["mem"].next = mem_init

    max_cycles = 1000
    while not outputs["halted"].val and outputs["cycle"].val < max_cycles:
        sim.step()
    
    actual_c = []
    for r in range(N):
        row = []
        for c in range(N):
            row.append(outputs["mem"].val[2 * N * N + r * N + c])
        actual_c.append(row)

    print("Actual:")
    for row in actual_c:
        print(row)

    expected = gemm(A, B, C, N)
    print("Expected:")
    for row in expected:
        print(row)
    
