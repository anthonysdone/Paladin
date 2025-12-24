import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import *
from enum import Enum
from dataclasses import dataclass
from typing import Optional

# === Ops and Branch Kinds ===

class AluOp(Enum):
    ADD   = 0
    SUB   = 1
    AND   = 2
    OR    = 3
    XOR   = 4
    SLL   = 5
    SRL   = 6
    SRA   = 7
    SLT   = 8
    SLTU  = 9
    ADDI  = 10
    LUI   = 11
    AUIPC = 12
    NOP   = 13

class BranchKind(Enum):
    NONE = 0
    BEQ  = 1
    BNE  = 2
    BLT  = 3
    BGE  = 4
    BLTU = 5
    BGEU = 6
    JAL  = 7
    JALR = 8

class MemOperation(Enum):
    NONE  = 0
    READ  = 1
    WRITE = 2

# === Structs ===

@dataclass
class FetchToDecode:
    instr: int
    pc: int

@dataclass
class DecodeToExec: 
    rd: Optional[int] = None
    op: AluOp = AluOp.NOP
    br: BranchKind = BranchKind.NONE
    mem: MemOperation = MemOperation.NONE
    imm: Optional[int] = None
    rs1_val: int = 0
    rs2_val: int = 0
    pc: int = 0

@dataclass
class ExecToMem: 
    addr_or_alu: int = 0
    store_data: int = 0
    wb_data_nonload: int = 0
    mem: MemOperation = MemOperation.NONE
    rd: Optional[int] = None

@dataclass
class MemToWb:
    wb_data: int = 0
    rd: Optional[int] = None

# === Helper Functions ===

def sext32(x, bits):
    m = 1 << (bits - 1)
    return (x ^ m) - m

def mask32(x):
    return x & 0xFFFFFFFF

# === IF ===

@task
def fetch_stage(pc, imem, stall_if, redirect_pc, fetch_to_decode):
    fetch_to_decode.next = None

    if redirect_pc.val is not None:
        pc.next = redirect_pc.val
    elif stall_if.val:
        pc.next = pc.val
    else:
        pc_addr = pc.val >> 2
        if pc_addr < len(imem.val):
            instr = imem.val[pc_addr]
            fetch_to_decode.next = FetchToDecode(instr=instr, pc=pc.val)
            pc.next = pc.val + 4
        else: 
            pc.next = pc.val

class DataHazardManager: 
    def __init__(self):
        self.scoreboard = 0
    
    def is_locked(self, r):
        return bool((self.scoreboard >> r) & 0x1)
    
    def lock_reg(self, r):
        self.scoreboard |= (0x1 << r)
    
    def release_reg(self, r):
        self.scoreboard ^= (0x1 << r)
    
    def copy(self): # was having issues with aliasing
        new = DataHazardManager()
        new.scoreboard = self.scoreboard
        return new

# === ID ===

@task
def decode_stage(fetch_to_decode_in, saved_fetch_to_decode, hazardManager, wb_finished, regfile, decode_to_exec, stall_request):
    stall_request.next = False
    decode_to_exec.next = None

    if wb_finished.val is not None: 
        hm = hazardManager.val.copy()
        hm.release_reg(wb_finished.val)
        hazardManager.next = hm
    else:
        hazardManager.next = hazardManager.val.copy()
    
    fetch_to_decode = fetch_to_decode_in.val
    if saved_fetch_to_decode.val is not None:
        fetch_to_decode = saved_fetch_to_decode.val
    saved_fetch_to_decode.next = None

    if fetch_to_decode is None:
        return 
    
    instr = fetch_to_decode.instr
    opcode = instr & 0x7f
    rd = (instr >> 7) & 0x1f
    funct3 = (instr >> 12) & 0x7
    rs1 = (instr >> 15) & 0x1f
    rs2 = (instr >> 20) & 0x1f
    funct7 = (instr >> 25) & 0x7f

    if hazardManager.val.is_locked(rs1) or hazardManager.val.is_locked(rs2):
        saved_fetch_to_decode.next = fetch_to_decode
        stall_request.next = True
        return
    
    dec = DecodeToExec(pc=fetch_to_decode.pc)

    def imm_i(): 
        return sext32(instr >> 20, 12)
    def imm_s():
        v = ((instr >> 7) & 0x1f) | (((instr >> 25) & 0x7f) << 5)
        return sext32(v, 12)
    def imm_b():
        v = (((instr >> 8) & 0x0f) << 1) | (((instr >> 25) & 0x3f) << 5) | (((instr >> 7) & 0x01) << 11) | (((instr >> 31) & 0x01) << 12)
        return sext32(v, 13)
    def imm_u():
        return mask32(instr & 0xfffff000)
    def imm_j():
        v = (((instr >> 21) & 0x3ff) << 1) | (((instr >> 20) & 0x001) << 11) | (((instr >> 12) & 0x0ff) << 12) | (((instr >> 31) & 0x001) << 20)
        return sext32(v, 21)

    dec.rs1_val = regfile.val[rs1]
    dec.rs2_val = regfile.val[rs2]

    if opcode == 0x33: 
        if rd != 0:
            dec.rd = rd
        key = (funct7) << 3 | funct3
        if key == (0x00 << 3) | 0x0: dec.op = AluOp.ADD
        elif key == (0x20 << 3) | 0x0: dec.op = AluOp.SUB
        elif key == (0x00 << 3) | 0x7: dec.op = AluOp.AND
        elif key == (0x00 << 3) | 0x6: dec.op = AluOp.OR
        elif key == (0x00 << 3) | 0x4: dec.op = AluOp.XOR
        elif key == (0x00 << 3) | 0x1: dec.op = AluOp.SLL
        elif key == (0x00 << 3) | 0x5: dec.op = AluOp.SRL
        elif key == (0x20 << 3) | 0x5: dec.op = AluOp.SRA
        elif key == (0x00 << 3) | 0x2: dec.op = AluOp.SLT
        elif key == (0x00 << 3) | 0x3: dec.op = AluOp.SLTU
        else: dec.op = AluOp.NOP
    
    elif opcode == 0x13:
        if rd != 0:
            dec.rd = rd
        dec.imm = imm_i()
        if funct3 == 0x0: dec.op = AluOp.ADDI
        elif funct3 == 0x2: dec.op = AluOp.SLT
        elif funct3 == 0x3: dec.op = AluOp.SLTU
        elif funct3 == 0x4: dec.op = AluOp.XOR
        elif funct3 == 0x6: dec.op = AluOp.OR
        elif funct3 == 0x7: dec.op = AluOp.AND
        elif funct3 == 0x1:
            if (funct7 & 0x7f) == 0x00: 
                dec.op = AluOp.SLL
            else: 
                dec.op = AluOp.NOP
                dec.rd = None
                dec.imm = None
        elif funct3 == 0x5: 
            if (funct7 & 0x20) == 0x20: 
                dec.op = AluOp.SRA
            else:
                dec.op = AluOp.NOP
                dec.rd = None
                dec.imm = None
        else:
            dec.rd = None
            dec.imm = None
        
    elif opcode == 0x03:
        if rd != 0:
            dec.rd = rd
        dec.imm = imm_i()
        dec.mem = MemOperation.READ
        dec.op = AluOp.ADD
    
    elif opcode == 0x23: 
        dec.imm = imm_s()
        dec.mem = MemOperation.WRITE
        dec.op = AluOp.ADD
    
    elif opcode == 0x63: 
        dec.imm = imm_b()

        if funct3 == 0x0: dec.br = BranchKind.BEQ
        elif funct3 == 0x1: dec.br = BranchKind.BNE
        elif funct3 == 0x4: dec.br = BranchKind.BLT
        elif funct3 == 0x5: dec.br = BranchKind.BGE
        elif funct3 == 0x6: dec.br = BranchKind.BLTU
        elif funct3 == 0x7: dec.br = BranchKind.BGEU

    elif opcode == 0x37:
        if rd != 0:
            dec.rd = rd
        dec.op = AluOp.LUI
        dec.imm = imm_u()
    
    elif opcode == 0x17:
        if rd != 0:
            dec.rd = rd
        dec.op = AluOp.AUIPC
        dec.imm = imm_u()
    
    elif opcode == 0x6f:
        if rd != 0:
            dec.rd = rd
        dec.br = BranchKind.JAL
        dec.imm = imm_j()
    
    elif opcode == 0x67:
        if rd != 0:
            dec.rd = rd
        dec.br = BranchKind.JALR
        dec.imm = imm_i()
    
    stall_request.next = (dec.br != BranchKind.NONE)

    if (dec.rd is not None):
        hm = hazardManager.val.copy()
        hm.lock_reg(dec.rd)
        hazardManager.next = hm
    
    decode_to_exec.next = dec

# === EX ===

@task
def execute_stage(decode_to_exec, exec_to_mem, redirect_pc): 
    redirect_pc.next = None
    exec_to_mem.next = None

    if decode_to_exec.val is None:
        return 
    
    dec = decode_to_exec.val
    em = ExecToMem()

    em.mem = dec.mem
    em.rd = dec.rd

    rs1_val = dec.rs1_val
    rs2_val = dec.rs2_val

    op2 = dec.imm if dec.imm is not None else rs2_val

    shamt5 = (dec.imm & 31) if dec.imm is not None else (rs2_val & 31)

    alu = 0
    def do_slt(a, b): 
        return 1 if a < b else 0
    def do_sltu(a, b): 
        return 1 if a < b else 0
    
    if dec.op == AluOp.ADD or dec.op == AluOp.ADDI:
        alu = mask32(rs1_val + op2)
    elif dec.op == AluOp.SUB:
        alu = mask32(rs1_val - rs2_val)
    elif dec.op == AluOp.AND: 
        alu = mask32(rs1_val & op2)
    elif dec.op == AluOp.OR:
        alu = mask32(rs1_val | op2)
    elif dec.op == AluOp.XOR:
        alu = mask32(rs1_val ^ op2)
    elif dec.op == AluOp.SRL:
        alu = mask32(rs1_val << shamt5)
    elif dec.op == AluOp.SRA:
        alu = mask32(int(rs1_val) >> shamt5) if int(rs1_val) < 0 else mask32(rs1_val >> shamt5) # having issues with int conversion
    elif dec.op == AluOp.SLT: 
        alu = do_slt(int(rs1_val), int(op2))
    elif dec.op == AluOp.SLTU:
        alu = do_sltu(rs1_val, op2)
    elif dec.op == AluOp.LUI: 
        alu = dec.imm if dec.imm is not None else 0
    elif dec.op == AluOp.AUIPC:
        alu = mask32(dec.pc + (dec.imm if dec.imm is not None else 0))
    else: 
        alu = 0
    
    link_val = 0
    def branch_target():
        return mask32(dec.pc + (dec.imm if dec.imm is not None else 0))
    
    if dec.br == BranchKind.BEQ: 
        if rs1_val == rs2_val:
            redirect_pc.next = branch_target()
    elif dec.br == BranchKind.BNE:
        if rs1_val != rs2_val: 
            redirect_pc.next = branch_target()
    elif dec.br == BranchKind.BLT:
        if int(rs1_val) < int(rs2_val):
            redirect_pc.next = branch_target()
    elif dec.br == BranchKind.BGE:
        if int(rs1_val) >= int(rs2_val):
            redirect_pc.next = branch_target()
    elif dec.br == BranchKind.BLTU:
        if rs1_val < rs2_val:
            redirect_pc.next = branch_target()
    elif dec.br == BranchKind.BGEU:
        if rs1_val >= rs2_val:
            redirect_pc.next = branch_target()
    elif dec.br == BranchKind.JAL:
        redirect_pc.next = branch_target()
        link_val = mask32(dec.pc + 4)
    elif dec.br == BranchKind.JALR:
        redirect_pc.next = mask32((rs1_val + (dec.imm if dec.imm is not None else 0)) & ~1)
        link_val = mask32(dec.pc + 4)
    
    if dec.br == BranchKind.JAL or dec.br == BranchKind.JALR:
        em.wb_data_nonload = link_val
    else:
        em.wb_data_nonload = alu
    
    em.addr_or_alu = alu
    em.store_data = rs2_val

    exec_to_mem.next = em

# === MEM ===

def mem_stage(exec_to_mem, dmem, mem_to_wb):
    mem_to_wb.next = None

    if exec_to_mem.val is None: 
        return 
    
    em = exec_to_mem.val
    mw = MemToWb()

    mw.rd = em.rd
    mw.wb_data = em.wb_data_nonload

    if em.mem == MemOperation.READ: 
        addr = em.addr_or_alu >> 2
        if addr < len(dmem.val):
            mw.wb_data = dmem.val[addr]
    elif em.mem == MemOperation.WRITE: 
        addr = em.addr_or_alu >> 2
        if addr < len(dmem.val):
            new_dmem = dmem.val.copy()
            new_dmem[addr] = em.store_data
            dmem.next = new_dmem
        
    mem_to_wb.next = mw

@task
def wb_stage(mem_to_wb, regfile, wb_finished):
    wb_finished.next = None

    if mem_to_wb.val is None:
        return 
    
    mw = mem_to_wb.val

    if mw.rd is None or mw.rd == 0:
        return

    if mw.rd is not None:
        new_regfile = regfile.val.copy()
        new_regfile[mw.rd] = mw.wb_data
        regfile.next = new_regfile