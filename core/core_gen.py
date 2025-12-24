import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import *
from core_tasks import *

def hex_byte(p, offset): 
    def hex_digit(c):
        if "0" <= c <= "9":
            return ord(c) - ord("0")
        elif "a" <= c <= "f":
            return 10 + ord(c) - ord("a")
        elif "A" <= c <= "F":
            return 10 + ord(c) - ord("A")
        else:
            raise RuntimeError(f"Invalid hex digit")
    hi = hex_digit(p[offset])
    lo = hex_digit(p[offset + 1])
    return (hi << 4) | lo

def load_hex(name, size):
    mem = [0] * size

    try:
        with open(name, "r") as f:
            base = 0
            saw_eof = False

            for lineno, line in enumerate(f, start=1):
                line = line.strip()

                if not line or not line.startswith(":"):
                    continue

                p = line[1:]
                n = len(p)

                def need(chars):
                    if 1 + chars > n:
                        raise RuntimeError(f"Line {lineno}: truncated")
                
                need(2)
                byte_count = hex_byte(p, 0)
                
                need(4)
                addr_hi_lo = (hex_byte(p, 2) << 8) | hex_byte(p, 4)

                need(2)
                rec_type = hex_byte(p, 6)

                data = []
                need(byte_count * 2 + 2)
                for i in range(byte_count):
                    data.append(hex_byte(p, 8 + i * 2))
                
                checksum = hex_byte(p, 8 + byte_count * 2)

                sum_val = byte_count + (addr_hi_lo >> 8) + (addr_hi_lo & 0xFF) + rec_type
                for b in data:
                    sum_val += b
                sum_val = (sum_val + checksum) & 0xFF

                if sum_val != 0:
                    raise RuntimeError(f"Line {lineno} checksum missmatch")
                
                if rec_type == 0x00:
                    addr = base + addr_hi_lo
                    for i, byte_val in enumerate(data):
                        a = addr + i
                        word_index = a >> 2
                        byte_lane = a & 0x3

                        if word_index >= size:
                            raise RuntimeError(f"Line {lineno}: write address out of range(0x{a:08x})")
                        
                        mem[word_index] &= ~(0xFF << (byte_lane * 8))
                        mem[word_index] |= (byte_val << (byte_lane * 8))

                elif rec_type == 0x01:
                    saw_eof = True
                elif rec_type == 0x02:
                    if byte_count != 2:
                        raise RuntimeError(f"Line {lineno}: ESA length must be 2")
                    seg = (data[0] << 8) | data[1]
                    base = seg << 4
                elif rec_type == 0x04:
                    if byte_count != 2:
                        raise RuntimeError(f"Line {lineno}: ESA length must be 2")
                    seg = (data[0] << 8) | data[1]
                    base = seg << 16
                
                elif rec_type == 0x03 or rec_type == 0x05:
                    pass
                else:
                    raise RuntimeError(f"Line {lineno}: unknown record type {rec_type:02x}")
            
            if not saw_eof:
                raise RuntimeError("HEX file missing EOF record (type 01)")
            
    except FileNotFoundError:
        raise Exception(f"Failed to open HEX file: {name}")

    return mem


def rv32i_5stage(program):
    sim = Sim()

    IMEM_SIZE = 1024
    DMEM_SIZE = 2048
    ARR_LEN = 16 

    imem_data = load_hex(program, IMEM_SIZE)
    imem = sim.reg(imem_data)

    dmem_data = [0] * DMEM_SIZE
    if DMEM_SIZE < (ARR_LEN * 2): 
        raise RuntimeError("DMEM too small for array")
    for i in range(ARR_LEN):
        dmem_data[i] = ARR_LEN - i
    dmem = sim.reg(dmem_data)

    regfile_data = [0] * 32
    regfile_data[2] = DMEM_SIZE - 4
    regfile = sim.reg(regfile_data)

    if_id_reg = sim.reg(None)
    id_ex_reg = sim.reg(None)
    ex_mem_reg = sim.reg(None)
    mem_wb_reg = sim.reg(None)

    stall_if = sim.reg(False)
    redirect_pc = sim.reg(None)
    wb_finished = sim.reg(None)

    # === IF ===

    pc = sim.reg(0)
    sim.add(fetch_stage(
        pc, 
        imem,
        stall_if,
        redirect_pc,
        if_id_reg
    ))

    # === ID ===
    
    saved_if_id = sim.reg(None)
    hazard_manager = sim.reg(DataHazardManager())
    sim.add(decode_stage(
        if_id_reg, 
        saved_if_id, 
        hazard_manager, 
        wb_finished, 
        regfile, 
        id_ex_reg, 
        stall_if
    ))

    # === EX ===

    sim.add(execute_stage(
        id_ex_reg, 
        ex_mem_reg,
        redirect_pc
    ))

    # === MEM ===

    sim.add(mem_stage(
        ex_mem_reg, 
        dmem, 
        mem_wb_reg
    ))

    # === WB ===

    sim.add(wb_stage(
        mem_wb_reg, 
        regfile,
        wb_finished
    ))

    # === Outputs ===

    outputs = {
        "pc": pc,
        "regfile": regfile,
        "dmem": dmem, 
        "if_id_reg": if_id_reg, 
        "id_ex_reg": id_ex_reg, 
        "ex_mem_reg": ex_mem_reg,
        "mem_wb_reg": mem_wb_reg, 
        "hazard_manager": hazard_manager
    }

    return sim, outputs

if __name__ == "__main__":
    num_cores = 1
    program = "software/bubblesort.hex"

    print(f"Instantiating core...")
    sim, outputs = rv32i_5stage(program)
    print(f"Running core simulation...")

    for cycle in range(2001): 
        sim.run(1)

        if cycle % 100 == 0:
            print(f"\nCycle {cycle}:")
            print(f"PC: 0x{outputs['pc'].val:08x}")
            print(f"x3: {outputs['regfile'].val[3]}")
            print(f"x4: {outputs['regfile'].val[4]}")
            print(f"x5: {outputs['regfile'].val[5]}")
            print(f"DMem[0:4]: {outputs['dmem'].val}")