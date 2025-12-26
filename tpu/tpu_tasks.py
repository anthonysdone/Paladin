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
    # DMAA = 2
    # DMAB = 3
    # DMAC = 4
    # ZERO = 5
    # EXEC = 6
    # COMM = 7
    # DMAO = 8
    INIT = 2
    LDC  = 3
    LOOP = 4
    LDA  = 5
    LDB  = 6
    EXEC = 7
    COMM = 8
    KNXT = 9
    STC = 10
    MNXT = 11
    NEXT = 12
    HALT = 13

class DmaKind(Enum):
    NONE = 0
    LDA  = 1
    LDB  = 2
    LDC  = 3
    STC  = 4

class OpKind(Enum):
    MNK  = 0
    TILE = 1
    GEMM = 2
    HALT = 3

# === Structs ===

@dataclass
class TpuOp:
    kind: OpKind
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0

# this wouldnt be in the actual implementation but im using python instead of actual asm :P
def parse_instr(instr):
    op = instr[0].lower()

    match op:
        case "mnk":
            return TpuOp(OpKind.MNK, int(instr[1]), int(instr[2]), int(instr[3]))
        case "tile":
            return TpuOp(OpKind.TILE, int(instr[1]))
        case "gemm":
            return TpuOp(OpKind.GEMM)
        case "halt":
            return TpuOp(OpKind.HALT)
        case _:
            raise RuntimeError(f"Unknown instruction {op}")

# === DMA ===

@task
def dma(state, kind, base, step, r, c, done, t_rows, t_cols, mem, spad_a, spad_b, sums_c, T):
    cur_state = state.val

    if cur_state not in [TpuState.LDA, TpuState.LDB, TpuState.LDC, TpuState.STC]:
        done.next = False
        return 

    if kind.val == DmaKind.NONE:
        done.next = False
        return

    # Check if kind has changed (new DMA operation started)
    # We do this by checking if r and c are both 0, which indicates we just initialized
    if r.val == 0 and c.val == 0 and done.val:
        # This shouldn't happen - r and c shouldn't be 0 if we're in the middle of DMA
        # So if they are, done should be false
        done.next = False
        return

    if done.val:
        # Already done, keep done high until kind clears
        return

    mem_addr = base.val + r.val * step.val + c.val
    spad_idx = r.val * T.val + c.val

    in_bounds = r.val < t_rows.val and c.val < t_cols.val

    if kind.val == DmaKind.LDA:
        new_spad_a = spad_a.val.copy()
        if in_bounds and mem_addr < len(mem.val):
            new_spad_a[spad_idx] = mem.val[mem_addr]
        else: 
            new_spad_a[spad_idx] = 0
        spad_a.next = new_spad_a
    
    elif kind.val == DmaKind.LDB:
        new_spad_b = spad_b.val.copy()
        if in_bounds and mem_addr < len(mem.val):
            new_spad_b[spad_idx] = mem.val[mem_addr]
        else:
            new_spad_b[spad_idx] = 0
        spad_b.next = new_spad_b
    
    elif kind.val == DmaKind.LDC:
        new_sums_c = sums_c.val.copy()
        if in_bounds and mem_addr < len(mem.val):
            new_sums_c[spad_idx] = mem.val[mem_addr]
        else:
            new_sums_c[spad_idx] = 0
        sums_c.next = new_sums_c
    
    elif kind.val == DmaKind.STC:
        new_mem = mem.val.copy()
        if in_bounds and mem_addr < len(mem.val):
            new_mem[mem_addr] = sums_c.val[spad_idx]
        mem.next = new_mem
    
    next_c = c.val + 1
    next_r = r.val
    dne = False

    if next_c >= t_cols.val:
        next_c = 0
        next_r = r.val + 1
        if next_r >= t_rows.val:
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
def feed_a_row(cycle, row_idx, spad_a, a_out, T):
    if cycle.val >= row_idx and cycle.val < row_idx + T:
        col = cycle.val - row_idx
        idx = row_idx * T + col
        a_out.next = spad_a.val[idx]
    else:
        a_out.next = 0

@task
def feed_b_col(cycle, col_idx, spad_b, b_out, T):
    if cycle.val >= col_idx and cycle.val < col_idx + T:
        row = cycle.val - col_idx
        idx = row * T + col_idx
        b_out.next = spad_b.val[idx]
    else:
        b_out.next = 0

@task
def s_counter(state, cycle):
    if state.val == TpuState.EXEC:
        cycle.next = cycle.val + 1

@task 
def commit(state, s_sums_flat, sums_c, T):
    if state.val == TpuState.COMM:
        new_sums_c = sums_c.val.copy()
        for i in range(T.val * T.val):
            new_sums_c[i] = new_sums_c[i] + s_sums_flat[i].val
        sums_c.next = new_sums_c

# === Controller ===

@task
def controller( # implemented as a fsm, idk if this will fly in lotus
    c_state, c_pc, c_op, c_prog, c_halt, c_count,
    d_kind, d_base, d_step, d_r, d_c, d_done, d_reads, d_writes,
    s_cycle, s_sums, s_spad_c, s_count, s_active, 
    t_rM, t_rK, t_rN, t_rT, t_m0, t_n0, t_k0, t_rows, t_cols, T,
    m_abase, m_bbase, m_cbase, m_astep, m_bstep, m_cstep
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
                    t_rM.next = c_op.val.arg0
                    t_rK.next = c_op.val.arg1
                    t_rN.next = c_op.val.arg2

                    m_abase.next = m_abase.val
                    m_astep.next = c_op.val.arg1
                    m_bbase.next = m_abase.val + c_op.val.arg0 * c_op.val.arg1
                    m_bstep.next = c_op.val.arg2
                    m_cbase.next = m_abase.val + c_op.val.arg0 * c_op.val.arg1 + c_op.val.arg1 * c_op.val.arg2
                    m_cstep.next = c_op.val.arg2
                    c_state.next = TpuState.NEXT

                case OpKind.TILE:
                    t_rT.next = c_op.val.arg0
                    T.next = c_op.val.arg0
                    c_state.next = TpuState.NEXT
                
                case OpKind.GEMM:
                    t_m0.next = 0
                    t_n0.next = 0
                    c_state.next = TpuState.INIT
                
                # case OpKind.STC:
                #     d_kind.next = DmaKind.STC
                #     d_base.next = c_op.val.arg0
                #     d_step.next = c_op.val.arg1
                #     d_r.next = 0
                #     d_c.next = 0
                #     c_state.next = TpuState.DMAO
                
                case OpKind.HALT:
                    c_state.next = TpuState.HALT
                    c_halt.next = True

        case TpuState.INIT:
            t_rows.next = min(t_rT.val, t_rM.val - t_m0.val)
            t_cols.next = min(t_rT.val, t_rN.val - t_n0.val)
            t_k0.next = 0
            # Reset systolic partial sums for this M,N tile
            for i in range(len(s_sums)):
                s_sums[i].next = 0
            c_state.next = TpuState.LDC

        case TpuState.LDC:
            if d_kind.val == DmaKind.NONE:
                # First entry to LDC state - initialize DMA
                d_kind.next = DmaKind.LDC
                d_base.next = m_cbase.val + t_m0.val * m_cstep.val + t_n0.val
                d_step.next = m_cstep.val
                d_r.next = 0
                d_c.next = 0
            elif d_done.val:
                # DMA completed
                d_kind.next = DmaKind.NONE
                d_r.next = 0
                d_c.next = 0
                d_reads.next = d_reads.val + t_rT.val * t_rT.val
                c_state.next = TpuState.LOOP

        case TpuState.LOOP:
            # Don't do anything - wait for the previous DMA to complete if needed
            # Check if we're waiting for LDC DMA to complete (which happens from INIT->LDC->LOOP)
            if d_kind.val == DmaKind.LDC and d_done.val:
                # LDC DMA completed, now start next iteration
                if t_k0.val >= t_rK.val:
                    d_kind.next = DmaKind.NONE
                    c_state.next = TpuState.STC
                else:
                    # Resetting systolic array for next K iteration
                    for i in range(len(s_sums)):
                        s_sums[i].next = 0
                    t_rows.next = min(t_rT.val, t_rM.val - t_m0.val)
                    t_cols.next = min(t_rT.val, t_rN.val - t_n0.val)
                    c_state.next = TpuState.LDA
            elif d_kind.val == DmaKind.NONE:
                # We're coming from KNXT, reset systolic array for next K iteration
                for i in range(len(s_sums)):
                    s_sums[i].next = 0
                if t_k0.val >= t_rK.val:
                    c_state.next = TpuState.STC
                else:
                    t_rows.next = min(t_rT.val, t_rM.val - t_m0.val)
                    t_cols.next = min(t_rT.val, t_rN.val - t_n0.val)
                    c_state.next = TpuState.LDA
        
        case TpuState.LDA:
            if d_kind.val == DmaKind.NONE:
                # First entry to LDA state - initialize DMA
                d_kind.next = DmaKind.LDA
                d_base.next = m_abase.val + (t_m0.val * m_astep.val) + t_k0.val
                d_step.next = m_astep.val
                d_r.next = 0
                d_c.next = 0
            elif d_done.val:
                # DMA completed
                d_kind.next = DmaKind.NONE
                d_r.next = 0
                d_c.next = 0
                d_reads.next = d_reads.val + t_rT.val * t_rT.val
                c_state.next = TpuState.LDB
        
        case TpuState.LDB:
            if d_kind.val == DmaKind.NONE:
                # First entry to LDB state - initialize DMA
                d_kind.next = DmaKind.LDB
                d_base.next = m_bbase.val + (t_k0.val * m_bstep.val) + t_n0.val
                d_step.next = m_bstep.val
                d_r.next = 0
                d_c.next = 0
            elif d_done.val:
                # DMA completed
                d_kind.next = DmaKind.NONE
                d_r.next = 0
                d_c.next = 0
                d_reads.next = d_reads.val + t_rT.val * t_rT.val
                t_rows.next = min(t_rT.val, t_rK.val - t_k0.val)
                t_cols.next = min(t_rT.val, t_rN.val - t_n0.val)
                s_cycle.next = 0
                c_state.next = TpuState.EXEC


        case TpuState.EXEC:
            if d_done.val:
                d_kind.next = DmaKind.NONE
                d_reads.next = d_reads.val + t_rT.val * t_rT.val
            
            s_active.next = True
            if s_cycle.val >= 3 * t_rT.val - 2:
                s_active.next = False
                s_count.next = s_count.val + t_rT.val * t_rT.val
                c_state.next = TpuState.COMM
        
        case TpuState.COMM:
            c_state.next = TpuState.KNXT
        
        case TpuState.KNXT:
            t_k0.next = t_k0.val + t_rT.val
            c_state.next = TpuState.LOOP
        
        case TpuState.STC:
            if d_kind.val == DmaKind.NONE:
                # First entry to STC state - initialize DMA
                t_rows.next = min(t_rT.val, t_rM.val - t_m0.val)
                t_cols.next = min(t_rT.val, t_rN.val - t_n0.val)
                d_kind.next = DmaKind.STC
                d_base.next = m_cbase.val + t_m0.val * m_cstep.val + t_n0.val
                d_step.next = m_cstep.val
                d_r.next = 0
                d_c.next = 0
            elif d_done.val:
                # DMA completed
                d_kind.next = DmaKind.NONE
                d_r.next = 0
                d_c.next = 0
                d_writes.next = d_writes.val + t_rT.val * t_rT.val
                c_state.next = TpuState.MNXT
        
        case TpuState.MNXT:
            if d_done.val:
                d_kind.next = DmaKind.NONE
                d_writes.next = d_writes.val + t_rT.val * t_rT.val
            
            n0_next = t_n0.val + t_rT.val
            m0_next = t_m0.val

            if n0_next >= t_rN.val:
                n0_next = 0
                m0_next = t_m0.val + t_rT.val

            t_n0.next = n0_next
            t_m0.next = m0_next

            if m0_next >= t_rM.val:
                c_state.next = TpuState.NEXT
            else:
                c_state.next = TpuState.INIT

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
