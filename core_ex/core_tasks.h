
#include <array>
#include <cstdint>
#include <cstdio>

#include "ash_types.h"

#include <optional>
#include <tuple>
#include <bit>

using std::optional;
using std::tuple;
using std::array;

// #define PRINTF(fmt, ...) \
//     do { \
//         printf(fmt, ##__VA_ARGS__); \
//         fflush(stdout); \
//     } while (0)

// #define CORE_PRINTF(fmt, ...) PRINTF(fmt, ##__VA_ARGS__)


#define PRINTF(fmt, ...)
#define CORE_PRINTF(fmt, ...)

#ifdef __riscv
    #define ASSERT(x) if (!(x)) ash::simFail();
#else
    #define ASSERT(x) assert(x);
#endif

// ---- Opaque-ish ops and branch kinds (internal) -----------------------------
enum class AluOp : uint8_t {
    ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU, ADDI, LUI, AUIPC, NOP
};
enum class BranchKind : uint8_t { NONE, BEQ, BNE, BLT, BGE, BLTU, BGEU, JAL, JALR };

enum class MemOperation : uint8_t { NONE, READ, WRITE };

static inline const char* to_cstr(AluOp op) {
    switch (op) {
        case AluOp::ADD:   return "ADD";
        case AluOp::SUB:   return "SUB";
        case AluOp::AND:   return "AND";
        case AluOp::OR:    return "OR";
        case AluOp::XOR:   return "XOR";
        case AluOp::SLL:   return "SLL";
        case AluOp::SRL:   return "SRL";
        case AluOp::SRA:   return "SRA";
        case AluOp::SLT:   return "SLT";
        case AluOp::SLTU:  return "SLTU";
        case AluOp::ADDI:  return "ADDI";
        case AluOp::LUI:   return "LUI";
        case AluOp::AUIPC: return "AUIPC";
        case AluOp::NOP:   return "NOP";
    }
    return "?";
}

static inline const char* to_cstr(BranchKind b) {
    switch (b) {
        case BranchKind::NONE: return "NONE";
        case BranchKind::BEQ:  return "BEQ";
        case BranchKind::BNE:  return "BNE";
        case BranchKind::BLT:  return "BLT";
        case BranchKind::BGE:  return "BGE";
        case BranchKind::BLTU:  return "BLTU";
        case BranchKind::BGEU:  return "BGEU";
        case BranchKind::JAL:  return "JAL";
        case BranchKind::JALR: return "JALR";
    }
    return "?";
}

static inline const char* to_cstr(MemOperation m) {
    switch (m) {
        case MemOperation::NONE:  return "NONE";
        case MemOperation::READ:  return "READ";
        case MemOperation::WRITE: return "WRITE";
    }
    return "?";
}

// ---- optional printers ------------------------------------------------------

// For optionals whose T has a .print()
template <typename T>
inline void print_optional(const optional<T> op) {
    if (op) {
        op->print();
    } else {
        CORE_PRINTF("(none)");
    }
}

// For scalar optionals (e.g., uint8_t, Word)
template <typename T>
static inline void print_opt_scalar(const char* name, const optional<T>& v) {
    if (v) {
        // uint32_t cast to print as unsigned
        CORE_PRINTF("%s=%u", name, static_cast<unsigned>(*v));
    } else {
        CORE_PRINTF("%s=none", name);
    }
}

template <>
inline void print_opt_scalar<Word>(const char* name, const optional<Word>& v) {
    if (v) {
        CORE_PRINTF("%s=0x%08x", name, *v);
    } else {
        CORE_PRINTF("%s=none", name);
    }
}

// ---- structs ---------------------------------------------------------------

struct FetchToDecode {
    Word instr;
    Word pc;

    void print() const {
        CORE_PRINTF("FetchToDecode{instr=0x%08x, pc=0x%08x}", instr, pc);
    }
};

struct DecodeToExec {
    optional<uint8_t> rd  = std::nullopt;
    AluOp       op        = AluOp::NOP;
    BranchKind  br        = BranchKind::NONE; // BEQ/BNE/JAL/JALR detection
    MemOperation mem      = MemOperation::NONE;
    optional<Word> imm    = std::nullopt;
    Word          rs1_val = 0;
    Word          rs2_val = 0;
    Word          pc      = 0;

    void print() const {
        CORE_PRINTF("DecodeToExec{");
        print_opt_scalar("rd", rd);
        CORE_PRINTF(", op=%s, br=%s, mem=%s", to_cstr(op), to_cstr(br), to_cstr(mem));
        CORE_PRINTF(", ");
        print_opt_scalar<Word>("imm", imm);
        CORE_PRINTF(", rs1=0x%08x, rs2=0x%08x, pc=0x%08x", rs1_val, rs2_val, pc);
        CORE_PRINTF("}");
    }
};

struct ExecToMem {
    Word addr_or_alu     = 0;
    Word store_data      = 0;
    Word wb_data_nonload = 0;
    MemOperation mem     = MemOperation::NONE;
    optional<uint8_t> rd = std::nullopt;

    void print() const {
        CORE_PRINTF("ExecToMem{addr_or_alu=0x%08x, store_data=0x%08x, wb_data_nonload=0x%08x",
                    addr_or_alu, store_data, wb_data_nonload);
        CORE_PRINTF(", mem=%s, ", to_cstr(mem));
        print_opt_scalar("rd", rd);
        CORE_PRINTF("}");
    }
};

struct MemToWb {
    Word              wb_data = 0;
    optional<uint8_t> rd      = std::nullopt;

    void print() const {
        CORE_PRINTF("MemToWb{wb_data=0x%08x, ", wb_data);
        print_opt_scalar("rd", rd);
        CORE_PRINTF("}");
    }
};

constexpr int IMemSize = 1024;
constexpr int DMemSize = 2048;

// Memories
using RegFile = std::array<Word, 32>;
using IMem    = std::array<Word, IMemSize>;
using DMem    = std::array<Word, DMemSize>;

// Helpers
static inline Word sext32(Word x, int bits) {
    const uint32_t m = 1u << (bits - 1);
    return (x ^ m) - m;
}


// ===================== IF (Instruction Fetch) =====================
// Inputs:
// - pc_in .................. from IF.pc_out (self-feedback)
// - imem ................... memory object (not an edge)
// - stall_if ............... from ID.stall_request  (single source: ID)
// - redirect_pc ............ from EX.redirect_pc    (single source: EX)
//
// Outputs (consumers):
// - instr_out .............. → ID.instr_in
// - pc_out ................. → IF.pc_in (self-feedback), → ID.pc_for_id
// - valid_out .............. → ID.valid_in
TASK
fetch_stage(ARG(INOUT, FULL, Word) pc,
            ARG(IN, PARTIAL, IMem) imem,
            ARG(IN, FULL, bool) stall_if,
            ARG(IN, FULL, optional<Word>) redirect_pc,
            ARG(OUT, FULL, optional<FetchToDecode>) fetch_to_decode
            )
{
    fetch_to_decode = std::nullopt;

    CORE_PRINTF("[fetch_stage] pc=0x%x, stall_if=%d, ", pc, stall_if);
    print_opt_scalar<Word>("redirect_pc", redirect_pc);
    CORE_PRINTF("\n");

    if (redirect_pc.has_value()) {
        // Resolve to target and insert a bubble this cycle.
        pc = *redirect_pc;
    } else if (stall_if) {
        // Hold PC and emit bubble.
        // pc = pc;
    } else {
        // Normal fetch
        ASSERT(pc < imem.size() && "PC out of imem bounds (demo array)");
        Word instr_out = imem[pc>>2];
        PRINTF("fetch_stage: imem[%x] = %x\n", pc, instr_out);
        fetch_to_decode = FetchToDecode{.instr=instr_out, .pc=pc};
        pc        = pc + 4;
    }
}

class DataHazardManager {
    private:
        // dsm: One bit per register
        uint32_t scoreboard;

    public:
        DataHazardManager() {}

        void print() {
            printf("DataHazardManager [");
            for (uint32_t i = 0; i < 32; i++) {
                if ((scoreboard >> i) & 0x1) {
                    printf(" %u ", i);
                }
            }
            printf("]");
        }

        bool is_locked(uint8_t r) {
            // printf("DatahazardManger::is_locked(rd=%u) -- ", r); this->print(); printf("\n");
            return (scoreboard >> r) & 0x1;
        }

        void lock_reg(uint8_t r) {
            // printf("DatahazardManger::lock_reg(rd=%u) -- ", r); this->print(); printf("\n");
            //ASSERT(!is_locked(r));
            scoreboard |= (0x1 << r);
        }

        void release_reg(uint8_t r) {
            // printf("DatahazardManger::release_reg(rd=%u) -- ", r); this->print(); printf("\n");
            //ASSERT(is_locked(r));
            scoreboard ^= (0x1 << r);
        }
};

// ===================== ID (Decode / Reg Read) =====================
// Inputs:
// - instr_in ................ from IF.instr_out
// - pc_for_id .............. from IF.pc_out
// - valid_in ............... from IF.valid_out
// - regfile ................ memory object (not an edge)
//
// Outputs (consumers):
// - dec_out ................. → EX.dec_in
// - rs1_val ................. → EX.rs1_val
// - rs2_val ................. → EX.rs2_val
// - pc_to_ex ................ → EX.pc_ex
// - valid_out ............... → EX.valid_in
// - stall_request ........... → IF.stall_if  (assert when a branch/jump is detected)
// NOTE: single-source rule: stall_request is *the* source of IF.stall_if
TASK
decode_stage(ARG(IN, FULL, optional<FetchToDecode>) fetch_to_decode_in,
             ARG(INOUT, FULL, optional<FetchToDecode>) saved_fetch_to_decode,
             ARG(INOUT, PARTIAL, DataHazardManager)       hazardManager,
             ARG(IN, FULL, optional<uint8_t>)          wb_finished,
             ARG(IN,  PARTIAL, RegFile) regfile,
             ARG(OUT, FULL, optional<DecodeToExec>) decode_to_exec,
             ARG(OUT, FULL, bool)    stall_request)
{
    // Output defaults
    stall_request = false;
    decode_to_exec = std::nullopt;

    if (wb_finished.has_value()) hazardManager.release_reg(*wb_finished);

    optional<FetchToDecode> fetch_to_decode = fetch_to_decode_in;
    if (saved_fetch_to_decode.has_value()) {
        fetch_to_decode = saved_fetch_to_decode;
    }
    saved_fetch_to_decode.reset();

    CORE_PRINTF("[decode_stage] fetch_to_decode = ");
    print_optional(fetch_to_decode);
    CORE_PRINTF("\n");

    if (!fetch_to_decode.has_value()) return;


    const uint32_t instr  = fetch_to_decode->instr;
    const uint32_t opcode = instr & 0x7f;
    const uint32_t rd     = (instr >> 7)  & 0x1f;
    const uint32_t funct3 = (instr >> 12) & 0x7;
    const uint32_t rs1    = (instr >> 15) & 0x1f;
    const uint32_t rs2    = (instr >> 20) & 0x1f;
    const uint32_t funct7 = (instr >> 25) & 0x7f;

    if (hazardManager.is_locked(rs1) || hazardManager.is_locked(rs2)) {
        saved_fetch_to_decode = fetch_to_decode;
        stall_request = true;
        return;
    }


    decode_to_exec.emplace();

    decode_to_exec->pc      = fetch_to_decode->pc;

    auto imm_i = [&]{ return (Word)sext32(instr >> 20, 12); };
    auto imm_s = [&]{
        uint32_t v = ((instr >> 7) & 0x1f) | (((instr >> 25) & 0x7f) << 5);
        return (Word)sext32(v, 12);
    };
    auto imm_b = [&]{
        uint32_t v = (((instr >> 8)  & 0x0f) << 1)  |   // imm[4:1]
                     (((instr >> 25) & 0x3f) << 5)  |   // imm[10:5]
                     (((instr >> 7)  & 0x01) << 11) |   // imm[11]
                     (((instr >> 31) & 0x01) << 12);    // imm[12]
        return (Word)sext32(v, 13);
    };
    auto imm_u = [&]{ return (Word)(instr & 0xfffff000u); };
    auto imm_j = [&]{
        uint32_t v = (((instr >> 21) & 0x3ff) << 1)  |  // imm[10:1]
                     (((instr >> 20) & 0x001) << 11) |  // imm[11]
                     (((instr >> 12) & 0x0ff) << 12) |  // imm[19:12]
                     (((instr >> 31) & 0x001) << 20);   // imm[20]
        return (Word)sext32(v, 21);
    };

    // Operand reads (harmless if unused later)
    decode_to_exec->rs1_val = regfile[rs1];
    decode_to_exec->rs2_val = regfile[rs2];

    // --- Decode subset: R/I-ALU, LW, SW, BEQ/BNE, LUI, AUIPC, JAL/JALR ---
    switch (opcode) {
        case 0x33: { // R-type
            if (rd != 0) {
                decode_to_exec->rd = rd;
            }
            switch ((funct7 << 3) | funct3) {
                case (0x00<<3)|0x0: decode_to_exec->op = AluOp::ADD; break;
                case (0x20<<3)|0x0: decode_to_exec->op = AluOp::SUB; break;
                case (0x00<<3)|0x7: decode_to_exec->op = AluOp::AND; break;
                case (0x00<<3)|0x6: decode_to_exec->op = AluOp::OR;  break;
                case (0x00<<3)|0x4: decode_to_exec->op = AluOp::XOR; break;
                case (0x00<<3)|0x1: decode_to_exec->op = AluOp::SLL; break;
                case (0x00<<3)|0x5: decode_to_exec->op = AluOp::SRL; break;
                case (0x20<<3)|0x5: decode_to_exec->op = AluOp::SRA; break;
                case (0x00<<3)|0x2: decode_to_exec->op = AluOp::SLT; break;
                case (0x00<<3)|0x3: decode_to_exec->op = AluOp::SLTU;break;
                default:            decode_to_exec->op = AluOp::NOP; decode_to_exec->rd = std::nullopt; break;
            }
            break;
        }

        case 0x13: { // I-type ALU (ADDI/SLTI/SLTIU/SLLI/SRLI/SRAI/ANDI/ORI/XORI)
            if (rd != 0) {
                decode_to_exec->rd = rd;
            }
            decode_to_exec->imm     = imm_i();
            switch (funct3) {
                case 0x0: decode_to_exec->op = AluOp::ADDI; break; // ADDI
                case 0x2: decode_to_exec->op = AluOp::SLT;  break; // SLTI
                case 0x3: decode_to_exec->op = AluOp::SLTU; break; // SLTIU
                case 0x4: decode_to_exec->op = AluOp::XOR;  break;
                case 0x6: decode_to_exec->op = AluOp::OR;   break;
                case 0x7: decode_to_exec->op = AluOp::AND;  break;
                case 0x1: // SLLI (funct7 must be 0x00)
                    if ((funct7 & 0x7f) == 0x00) decode_to_exec->op = AluOp::SLL;
                    else { decode_to_exec->op = AluOp::NOP; decode_to_exec->rd = std::nullopt; decode_to_exec->imm = std::nullopt; }
                    break;
                case 0x5: // SRLI/SRAI distinguished by bit 30
                    if ((funct7 & 0x20) == 0x20) decode_to_exec->op = AluOp::SRA; else decode_to_exec->op = AluOp::SRL;
                    break;
                default:
                    decode_to_exec->op = AluOp::NOP; decode_to_exec->rd = std::nullopt; decode_to_exec->imm = std::nullopt; break;
            }
            break;
        }

        case 0x03: { // Loads (treat as LW for now)
            if (rd != 0) {
                decode_to_exec->rd = rd;
            }
            decode_to_exec->imm       = imm_i();
            decode_to_exec->mem       = MemOperation::READ;
            decode_to_exec->op        = AluOp::ADD; // addr = rs1 + imm
            break;
        }

        case 0x23: { // Stores (treat as SW for now)
            decode_to_exec->imm       = imm_s();
            decode_to_exec->mem       = MemOperation::WRITE;
            decode_to_exec->op        = AluOp::ADD; // addr = rs1 + imm
            break;
        }

        case 0x63: { // Branches
            // imm first (used regardless of kind)
            decode_to_exec->imm = imm_b();

            switch (funct3) {
                case 0x0: decode_to_exec->br = BranchKind::BEQ;  break;
                case 0x1: decode_to_exec->br = BranchKind::BNE;  break;
                case 0x4: decode_to_exec->br = BranchKind::BLT;  break;
                case 0x5: decode_to_exec->br = BranchKind::BGE;  break;
                case 0x6: decode_to_exec->br = BranchKind::BLTU; break;
                case 0x7: decode_to_exec->br = BranchKind::BGEU; break;
                default:  decode_to_exec->br = BranchKind::NONE; break; // (reserved/illegal)
            }
            break;
        }

        case 0x37: { // LUI
            if (rd != 0) {
                decode_to_exec->rd = rd;
            }
            decode_to_exec->op    = AluOp::LUI;
            decode_to_exec->imm   = imm_u();
            break;
        }

        case 0x17: { // AUIPC
            if (rd != 0) {
                decode_to_exec->rd = rd;
            }
            decode_to_exec->op    = AluOp::AUIPC;
            decode_to_exec->imm   = imm_u();
            break;
        }

        case 0x6f: { // JAL
            if (rd != 0) {
                decode_to_exec->rd = rd;
            }
            decode_to_exec->br    = BranchKind::JAL;
            decode_to_exec->imm   = imm_j();
            break;
        }

        case 0x67: { // JALR
            if (rd != 0) {
                decode_to_exec->rd = rd;
            }
            decode_to_exec->br    = BranchKind::JALR;
            decode_to_exec->imm   = imm_i();   // rs1 + imm in EX
            break;
        }

        default:
            // keep defaults (NOP)
            break;
    }

    // Control hazard policy:
    // - Stall earlier stages when a control-flow instr is in the pipe,
    //   but DO NOT block ID->EX for the current instruction.
    stall_request = (decode_to_exec->br != BranchKind::NONE);  // freeze IF (and ID input)

    if (decode_to_exec->rd.has_value()) {
        hazardManager.lock_reg(decode_to_exec->rd.value());
    }

}


// ===================== EX (Execute / Branch Resolve) =====================
// Inputs:
// - dec_in .................. from ID.dec_out
// - rs1_val ................. from ID.rs1_val
// - rs2_val ................. from ID.rs2_val
// - pc_ex ................... from ID.pc_to_ex
// - valid_in ................ from ID.valid_out
//
// Outputs (consumers):
// - alu_result .............. → MEM.addr_or_alu
// - store_data .............. → MEM.store_data
// - mem_read ................ → MEM.mem_read
// - mem_write ............... → MEM.mem_write
// - rd_out .................. → MEM.rd_in
// - wb_en_out ............... → MEM.wb_en_in
// - wb_data_nonload ......... → MEM.pass_through_wb_data
// - valid_out ............... → MEM.valid_in
// - redirect_valid .......... → IF.redirect_valid
// - redirect_pc ............. → IF.redirect_pc
TASK
execute_stage(ARG(IN, FULL, optional<DecodeToExec>) decode_to_exec,
              ARG(OUT, FULL, optional<ExecToMem>) exec_to_mem,
              ARG(OUT, FULL, optional<Word>) redirect_pc)
{
    redirect_pc = {};
    exec_to_mem = std::nullopt;

    CORE_PRINTF("[execute_stage] decode_to_exec = ");
    print_optional(decode_to_exec);
    CORE_PRINTF("\n");

    if (!decode_to_exec.has_value()) {
        return;
    }

    exec_to_mem.emplace();

    // Control + dest
    exec_to_mem->mem = decode_to_exec->mem;
    exec_to_mem->rd  = decode_to_exec->rd;

    Word rs1_val = decode_to_exec->rs1_val;
    Word rs2_val = decode_to_exec->rs2_val;

    // Operand selection
    const Word op2 = decode_to_exec->imm.has_value() ? decode_to_exec->imm.value() : rs2_val;

    // Shift amount: handle both reg and imm shift forms
    const uint32_t shamt5 = static_cast<uint32_t>(decode_to_exec->imm.has_value() ? (decode_to_exec->imm.value() & 31u) : (rs2_val & 31u));

    // ALU
    Word alu = 0;
    auto do_slt  = [](int32_t a, int32_t b) -> Word { return (a < b) ? 1u : 0u; };
    auto do_sltu = [](uint32_t a, uint32_t b) -> Word { return (a < b) ? 1u : 0u; };

    switch (decode_to_exec->op) {
        case AluOp::ADD:
        case AluOp::ADDI:  alu = rs1_val + op2; break;
        case AluOp::SUB:   alu = rs1_val - rs2_val; break;  // SUBI doesn't exist
        case AluOp::AND:   alu = rs1_val & op2; break;
        case AluOp::OR:    alu = rs1_val | op2; break;
        case AluOp::XOR:   alu = rs1_val ^ op2; break;
        case AluOp::SLL:   alu = rs1_val << shamt5; break;                  // SLL/SLLI
        case AluOp::SRL:   alu = rs1_val >> shamt5; break;                  // SRL/SRLI
        case AluOp::SRA:   alu = (Word)((int32_t)rs1_val >> shamt5); break; // SRA/SRAI
        case AluOp::SLT:   alu = do_slt ((int32_t)rs1_val, (int32_t)op2); break;  // uses op2
        case AluOp::SLTU:  alu = do_sltu(rs1_val, op2); break;                    // uses op2
        case AluOp::LUI:   alu = decode_to_exec->imm.value_or(0); break;
        case AluOp::AUIPC: alu = decode_to_exec->pc + decode_to_exec->imm.value_or(0); break;  // PC of instr + imm
        case AluOp::NOP:
        default:           alu = 0; break;
    }

    // Branch / jump resolution
    Word link_val = 0;

    auto branch_target = [&]() -> Word {
        // imm is already sign-extended in decode; value_or(0) is safe
        return decode_to_exec->pc + decode_to_exec->imm.value_or(0);
    };

    switch (decode_to_exec->br) {
        case BranchKind::BEQ:
            if (rs1_val == rs2_val) { redirect_pc = branch_target(); }
            break;

        case BranchKind::BNE:
            if (rs1_val != rs2_val) { redirect_pc = branch_target(); }
            break;

        // NEW: signed compares
        case BranchKind::BLT:
            if (static_cast<int32_t>(rs1_val) < static_cast<int32_t>(rs2_val)) {
                redirect_pc = branch_target();
            }
            break;

        case BranchKind::BGE:
            if (static_cast<int32_t>(rs1_val) >= static_cast<int32_t>(rs2_val)) {
                redirect_pc = branch_target();
            }
            break;

        // NEW: unsigned compares
        case BranchKind::BLTU:
            if (rs1_val < rs2_val) {  // both are Word (uint32_t)
                redirect_pc = branch_target();
            }
            break;

        case BranchKind::BGEU:
            if (rs1_val >= rs2_val) {
                redirect_pc = branch_target();
            }
            break;

        case BranchKind::JAL:
            redirect_pc = branch_target();               // target = PC + imm
            link_val    = decode_to_exec->pc + 4;        // rd gets link in WB
            break;

        case BranchKind::JALR:
            redirect_pc = (rs1_val + decode_to_exec->imm.value_or(0)) & ~1u; // LSB = 0
            link_val    = decode_to_exec->pc + 4;
            break;

        case BranchKind::NONE:
        default:
            break;
    }

    // MEM / WB data
    exec_to_mem->wb_data_nonload = (decode_to_exec->br == BranchKind::JAL || decode_to_exec->br == BranchKind::JALR) ? link_val : alu;
    exec_to_mem->addr_or_alu     = alu;         // used as address on loads/stores; ALU result otherwise
    exec_to_mem->store_data      = rs2_val;     // store value

}


// ===================== MEM (Data Memory) =====================
// Inputs:
// - valid_in ................ from EX.valid_out
// - addr_or_alu ............ from EX.alu_result
// - store_data .............. from EX.store_data
// - mem_read ................ from EX.mem_read
// - mem_write ............... from EX.mem_write
// - dmem .................... memory object (not an edge)
// - rd_in ................... from EX.rd_out
// - wb_en_in ................ from EX.wb_en_out
// - pass_through_wb_data .... from EX.wb_data_nonload
//
// Outputs (consumers):
// - wb_data ................. → WB.wb_data
// - rd_out .................. → WB.rd
// - wb_en_out ............... → WB.wb_en
// - valid_out ............... → WB.valid_in

enum class MemState : uint8_t { IDLE, WAITING_FOR_DCACHE };

TASK
mem_stage(ARG(IN, FULL, optional<ExecToMem>) exec_to_mem,
          ARG(INOUT, PARTIAL, DMem) dmem,
          ARG(OUT, FULL, optional<MemToWb>) mem_to_wb
          )
{

    mem_to_wb      = std::nullopt;

    CORE_PRINTF("[mem_stage] exec_to_mem = ");
    print_optional(exec_to_mem);
    CORE_PRINTF("\n");

    if (!exec_to_mem.has_value()) {
        return;
    }

    mem_to_wb.emplace();

    mem_to_wb->rd = exec_to_mem->rd;

    mem_to_wb->wb_data = exec_to_mem->wb_data_nonload;

    if (exec_to_mem->mem == MemOperation::READ) {
        Word addr = exec_to_mem->addr_or_alu >> 2; /*Divide by 4*/
        ASSERT(addr < dmem.size() && "DMEM read OOB (demo array)");
        mem_to_wb->wb_data = dmem[addr]; // LW
    }
    if (exec_to_mem->mem == MemOperation::WRITE) {
        Word addr = exec_to_mem->addr_or_alu >> 2; /*Divide by 4*/
        ASSERT(addr < dmem.size() && "DMEM write OOB (demo array)");
        dmem[addr] = exec_to_mem->store_data; // SW
    }
}


// ===================== WB (Write Back) =====================
// Inputs:
// - valid_in ................ from MEM.valid_out
// - wb_data ................. from MEM.wb_data
// - rd ...................... from MEM.rd_out
// - wb_en ................... from MEM.wb_en_out
// - regfile ................. memory object (not an edge)
//
TASK
wb_stage(ARG(IN, FULL, optional<MemToWb>) mem_to_wb,
         ARG(OUT, PARTIAL, RegFile) regfile,
         ARG(OUT, FULL, optional<uint8_t>) wb_finished
         )
{

    wb_finished = std::nullopt;

    CORE_PRINTF("[wb_stage] mem_to_wb = ");
    print_optional(mem_to_wb);
    CORE_PRINTF("\n");

    if (!mem_to_wb.has_value()) return;

    if (!mem_to_wb->rd.has_value()) return;

    if (mem_to_wb->rd.value() != 0) {
        wb_finished = mem_to_wb->rd.value();
        regfile[mem_to_wb->rd.value()] = mem_to_wb->wb_data;
    }
}