import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import *
from enum import Enum
from dataclasses import dataclass

# === States, Ops, Kinds ===

class TpuState(Enum):
    IF   = 0
    DEC  = 1
    DMAA = 2
    DMAB = 3
    DMAC = 4
    ZERO = 5
    EXEC = 6
    COMM = 7
    DMAO = 8
    NEXT = 9
    HALT = 9

class DmaKind(Enum):
    NONE = 0
    LDA  = 1
    LDB  = 2
    LDC  = 3
    STC  = 4

class OpKind(Enum):
    MNK = 0
    LDA = 1
    LDB = 2
    LDC = 3
    ZERO = 4
    GEMM = 5
    STC  = 6
    HALT = 7

# === Structs ===

@dataclass
class TpuOp:
    kind: OpKind
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0

# this wouldnt be in the actual implementation but im using python for asm :P
def parse_instr(instr):
    op = instr[0].lower()

    match op:
        case "mnk":
            return TpuOp(OpKind.MNK, int(instr[1]), int(instr[2]), int(instr[3]))
        case "lda":
            return TpuOp(OpKind.LDA, int(instr[1]), instr[2])
        case "ldb":
            return TpuOp(OpKind.LDB, int(instr[1]), instr[2])
        case "ldc":
            return TpuOp(OpKind.LDC, int(instr[1]), instr[2])
        case "zero":
            return TpuOp(OpKind.ZERO)
        case "gemm":
            return TpuOp(OpKind.GEMM)
        case "stc":
            return TpuOp(OpKind.STC, int(instr[1]), instr[2])
        case "halt":
            return TpuOp(OpKind.HALT)
        case _:
            raise RuntimeError(f"Unknown instruction {op}")

# === DMA ===

@task
def dma(state, kind, base, stride, r, c, done, mem, spad_a, spad_b, spad_c, N):
    cur_state = state.val

    if cur_state not in [TpuState.DMAA, TpuState.DMAB, TpuState.DMAC, TpuState.DMAO]:
        done.next = 0
        return 

    mem_addr = base.val + r.val * stride.val + c.val
    spad_idx = r.val * N + c.val

    if kind.val == DmaKind.LDA:
        new_spad_a = spad_a.val.copy()
        if mem_addr < len(mem.val):
            new_spad_a[spad_idx] = mem.val[mem_addr]
        spad_a.next = new_spad_a
    
    elif kind.val == DmaKind.LDB:
        new_spad_b = spad_b.val.copy()
        if mem_addr < len(mem.val):
            new_spad_b[spad_idx] = mem.val[mem_addr]
        spad_b.next = new_spad_b
    
    elif kind.val == DmaKind.LDC:
        new_spad_c = spad_c.val.copy()
        if mem_addr < len(mem.val):
            new_spad_c[spad_idx] = mem.val[mem_addr]
        spad_c.next = new_spad_c
    
    elif kind.val == DmaKind.STC:
        new_mem = mem.val.copy()
        if mem_addr < len(mem.val):
            new_mem[mem_addr] = spad_c.val[spad_idx]
        mem.next = new_mem
    
    next_c = c.val + 1
    next_r = r.val
    dne = False

    if next_c >= N:
        next_c = 0
        next_r = r.val + 1
        if next_r >= N:
            dne = True
            next_r = 0
    
    r.next = next_r
    c.next = next_c
    done.next = dne

# === Systolic ===

@task
def mac(a_in, b_in, sum, a_out, b_out, active):
    if active.val:
        sum.next = sum.val + a_in.val * b_in.val

    a_out.next = a_in.val
    b_out.next = b_in.val

@task
def feed_a_row(cycle, row_idx, spad_a, a_out, N):
    if cycle.val >= row_idx and cycle.val < row_idx + N:
        col = cycle.val - row_idx
        idx = row_idx * N + col
        a_out.next = spad_a.val[idx]
    else:
        a_out.next = 0

@task
def feed_b_col(cycle, col_idx, spad_b, b_out, N):
    if cycle.val >= col_idx and cycle.val < col_idx + N:
        row = cycle.val - col_idx
        idx = row * N + col_idx
        b_out.next = spad_b.val[idx]
    else:
        b_out.next = 0

@task
def s_counter(state, cycle):
    if state.val == TpuState.EXEC:
        cycle.next = cycle.val + 1

# === Controller ===

@task
def controller( # implemented as a fsm, idk if this will fly in lotus
    c_state, c_pc, c_op, c_prog, c_halt, c_count,
    d_kind, d_base, d_stride, d_r, d_c, d_done, d_reads, d_writes,
    s_cycle, s_sums, s_spad_c, s_count, s_active, N
):
    match c_state.val: 
        case TpuState.IF: 
            if c_pc.val < len(c_prog.val):
                op = parse_instr(c_prog.val[c_pc.val])
                c_op.next = op
                c_state.next = TpuState.DEC
            else:
                c_state.next = TpuState.HALT
                c_halt.next = True
        
        case TpuState.DEC: 
            match c_op.val.kind:
                case OpKind.MNK:
                    c_state.next = TpuState.NEXT
                
                case OpKind.LDA: 
                    d_kind.next = DmaKind.LDA
                    d_base.next = c_op.val.arg0
                    d_stride.next = c_op.val.arg1
                    d_r.next = 0
                    d_c.next = 0
                    c_state.next = TpuState.DMAA
                
                case OpKind.LDB:
                    d_kind.next = DmaKind.LDB
                    d_base.next = c_op.val.arg0
                    d_stride.next = c_op.val.arg1
                    d_r.next = 0
                    d_c.next = 0
                    c_state.next = TpuState.DMAB
                
                case OpKind.LDC:
                    d_kind.next = DmaKind.LDC
                    d_base.next = c_op.val.arg0
                    d_stride.next = c_op.val.arg1
                    d_r.next = 0
                    d_c.next = 0
                    c_state.next = TpuState.DMAC
                
                case OpKind.ZERO:
                    c_state.next = TpuState.ZERO
                
                case OpKind.GEMM:
                    s_cycle.next = 0
                    c_state.next = TpuState.EXEC
                
                case OpKind.STC:
                    d_kind.next = DmaKind.STC
                    d_base.next = c_op.val.arg0
                    d_stride.next = c_op.val.arg1
                    d_r.next = 0
                    d_c.next = 0
                    c_state.next = TpuState.DMAO
                
                case OpKind.HALT:
                    c_state.next = TpuState.HALT
                    c_halt.next = True
                
        case TpuState.DMAA:
            if d_done.val:
                d_kind.next = DmaKind.NONE
                d_reads.next = d_reads.val + N * N
                c_state.next = TpuState.NEXT
        
        case TpuState.DMAB:
            if d_done.val:
                d_kind.next = DmaKind.NONE
                d_reads.next = d_reads.val + N * N
                c_state.next = TpuState.NEXT
            
        case TpuState.DMAC:
            if d_done.val:
                d_kind.next = DmaKind.NONE
                d_reads.next = d_reads.val + N * N
                for i in range(N * N):
                    s_sums[i].next = s_spad_c.val[i]
                c_state.next = TpuState.NEXT
        
        case TpuState.ZERO:
            for i in range(N * N):
                s_sums[i].next = 0
            c_state.next = TpuState.NEXT
        
        case TpuState.EXEC:
            s_active.next = True
            if s_cycle.val >= 3 * N - 2:
                s_active.next = False
                s_count.next = s_count.val + N * N
                c_state.next = TpuState.COMM
        
        case TpuState.COMM:
            s_spad_c.next = [0] * (N * N)
            for i in range(N * N):
                s_spad_c.next[i] = s_sums[i].val
            c_state.next = TpuState.NEXT
        
        case TpuState.DMAO:
            if d_done.val:
                d_kind.next = DmaKind.NONE
                d_writes.next = d_writes.val + N * N
                c_state.next = TpuState.NEXT
        
        case TpuState.NEXT:
            c_pc.next = c_pc.val + 1
            c_count.next = c_count.val + 1
            c_state.next = TpuState.IF
        
        case TpuState.HALT:
            pass

@task
def c_counter(cycle, halted):
    if not halted.val:
        cycle.next = cycle.val + 1
