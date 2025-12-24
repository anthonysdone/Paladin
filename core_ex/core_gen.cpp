
#include <stdint.h>
#include <tuple>
#include <vector>
#include <bit>
#include <type_traits>
#include <assert.h>

#include <iostream>
#include <string_view>
#include <array>
#include <cstddef>
#include <cstring>

#include "taskflow.h"

#include "cores.h"


using namespace taskflow;

#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>

namespace detail {
    inline uint8_t hexByte(const char* p) {
        auto hex = [](char c)->int {
            if ('0' <= c && c <= '9') return c - '0';
            if ('a' <= c && c <= 'f') return 10 + (c - 'a');
            if ('A' <= c && c <= 'F') return 10 + (c - 'A');
            return -1;
        };
        int hi = hex(p[0]), lo = hex(p[1]);
        if (hi < 0 || lo < 0) throw std::runtime_error("Invalid hex digit");
        return static_cast<uint8_t>((hi << 4) | lo);
    }
}

/// Load an Intel HEX file into a little-endian array of 32-bit words.
/// - Data (type 00) records are written byte-wise at absolute addresses
///   (taking into account type 02/04 base), packed into 32-bit words as
///   little-endian: byte at addr k goes to (addr/4) word, byte lane (addr%4).
/// - Throws on malformed lines, checksum mismatch, or out-of-bounds writes.
/// - Uninitialized bytes remain 0.
template <int Size>
std::array<uint32_t, Size> load_hex(std::string name) {
    static_assert(Size > 0, "Size must be positive");

    std::array<uint32_t, Size> mem{};
    // zero-initialized already due to {}.

    std::ifstream in(name);
    if (!in) {
        throw std::runtime_error("Failed to open HEX file: " + name);
    }

    uint32_t base = 0; // upper address base (from type 02/04)
    bool saw_eof = false;

    std::string line;
    for (size_t lineno = 1; std::getline(in, line); ++lineno) {
        // Trim whitespace
        line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char c){ return c=='\r' || c=='\n' || c==' ' || c=='\t'; }), line.end());
        if (line.empty()) continue;

        if (line[0] != ':') {
            throw std::runtime_error("Line " + std::to_string(lineno) + ": missing ':'");
        }
        const char* p = line.c_str() + 1;
        const size_t n = line.size();

        auto need = [&](size_t chars){
            if (1 + chars > n) throw std::runtime_error("Line " + std::to_string(lineno) + ": truncated");
        };

        need(2);  uint8_t byteCount = detail::hexByte(p); p += 2;
        need(4);  uint16_t addrHiLo = (static_cast<uint16_t>(detail::hexByte(p)) << 8) | detail::hexByte(p+2); p += 4;
        need(2);  uint8_t recType = detail::hexByte(p); p += 2;

        // Read data bytes
        std::vector<uint8_t> data;
        data.reserve(byteCount);
        need(static_cast<size_t>(byteCount) * 2 + 2); // +2 for checksum at end
        for (int i = 0; i < byteCount; ++i) {
            data.push_back(detail::hexByte(p));
            p += 2;
        }
        // Checksum
        uint8_t checksum = detail::hexByte(p); p += 2;

        // Verify checksum: sum of (count, addr_hi, addr_lo, type, data...) + checksum == 0x00 mod 256
        uint32_t sum = byteCount + (addrHiLo >> 8) + (addrHiLo & 0xFF) + recType;
        for (uint8_t b : data) sum += b;
        sum = (sum + checksum) & 0xFF;
        if (sum != 0) {
            throw std::runtime_error("Line " + std::to_string(lineno) + ": checksum mismatch");
        }

        switch (recType) {
            case 0x00: { // data
                uint32_t addr = (base + static_cast<uint32_t>(addrHiLo));
                for (size_t i = 0; i < data.size(); ++i) {
                    uint32_t a = addr + static_cast<uint32_t>(i);
                    uint32_t wordIndex = a >> 2;      // a / 4
                    uint32_t byteLane  = a & 0x3;     // a % 4
                    if (wordIndex >= static_cast<uint32_t>(Size)) {
                        throw std::runtime_error("Line " + std::to_string(lineno) + ": write address out of range (0x" +
                                                  [&](){ std::ostringstream os; os<<std::hex<<std::uppercase<<a; return os.str(); }() + ")");
                    }
                    uint32_t v = mem[wordIndex];
                    // little-endian pack: lane 0 is LSB
                    v &= ~(0xFFu << (byteLane * 8));
                    v |= (static_cast<uint32_t>(data[i]) << (byteLane * 8));
                    mem[wordIndex] = v;
                }
                break;
            }
            case 0x01: { // EOF
                saw_eof = true;
                // Per spec, should be last; we can stop parsing further lines.
                // But keep reading to catch trailing junk if you prefer strictness.
                break;
            }
            case 0x02: { // Extended Segment Address (bits 4–19)
                if (byteCount != 2) throw std::runtime_error("Line " + std::to_string(lineno) + ": ESA length must be 2");
                uint16_t seg = (static_cast<uint16_t>(data[0]) << 8) | data[1];
                base = static_cast<uint32_t>(seg) << 4; // segment * 16
                break;
            }
            case 0x04: { // Extended Linear Address (upper 16 bits of 32-bit address)
                if (byteCount != 2) throw std::runtime_error("Line " + std::to_string(lineno) + ": ELA length must be 2");
                uint16_t upper = (static_cast<uint16_t>(data[0]) << 8) | data[1];
                base = static_cast<uint32_t>(upper) << 16;
                break;
            }
            case 0x03: // Start Segment Address (ignored here)
            case 0x05: // Start Linear Address (ignored here)
                // Valid records; ignore for raw memory load.
                break;
            default:
                throw std::runtime_error("Line " + std::to_string(lineno) + ": unknown record type " + std::to_string(recType));
        }

        if (saw_eof) break;
    }

    if (!saw_eof) {
        throw std::runtime_error("HEX file missing EOF record (type 01)");
    }

    return mem;
}

void rv32i_5stage() {
    // Memory Objects
    Reg<IMem> imem(load_hex<IMemSize>("quicksort.hex"));
    const uint32_t ARR_LEN = 1024;
    Reg<DMem> dmem([&ARR_LEN](){
        DMem dmem;
        assert(DMemSize >= (ARR_LEN*2)); // Space for stack
        for (uint32_t i = 0; i < ARR_LEN; i++) {
            dmem[i] = ARR_LEN-i;
        }
        return dmem;
    }());
    Reg<RegFile> regfile([&ARR_LEN](){
        RegFile regfile;
        regfile[2 /*sp*/] = DMemSize-4;
        return regfile;
    }());

    Reg<optional<FetchToDecode>> if_id_reg(std::nullopt);

    Reg<optional<DecodeToExec>> id_ex_reg(std::nullopt);

    Reg<optional<ExecToMem>>    ex_mem_reg(std::nullopt);

    Reg<optional<MemToWb>>      mem_wb_reg(std::nullopt);

    // ---------------- Single-cycle control wires --------
    Wire<bool> stall_if;                // ID.stall_request → IF.stall_if
    Wire<optional<Word>> redirect_pc;   // EX.redirect_pc → IF.redirect_pc

    //////
    Reg<optional<uint8_t>> wb_finished(std::nullopt);

    // ================== IF ==================
    {
        Reg<Word> pc(0);
        task(fetch_stage,
             pc,          // self-feedback
             imem,              // imem
             stall_if,          // from ID
             redirect_pc,       // from EX
             if_id_reg);      // to ID

    }

    // ================== ID ==================
    {
        Reg<optional<FetchToDecode>> saved_if_id(std::nullopt);
        Reg<DataHazardManager> hazardManager(DataHazardManager{});
        task(decode_stage,
             if_id_reg,
             saved_if_id,
             hazardManager,
             wb_finished,
             regfile,      // memory object
             id_ex_reg,
             stall_if      // → IF.stall_if (same cycle)
             );
    }

    // ================== EX ==================
    {
        task(execute_stage,
             id_ex_reg,
             ex_mem_reg,
             redirect_pc        // → IF.redirect_pc    (same cycle)
             );

    }

    // ================== MEM ==================

    {

        task(mem_stage,
            ex_mem_reg,
            dmem,              // memory object
            mem_wb_reg
            );

    }

    // ================== WB ==================
    {
        task(wb_stage,
             mem_wb_reg,
             regfile,
             wb_finished);
    }
}

int main(int argc, char* argv[]) {
    assert(argc == 2);
    uint32_t NCores = std::stoi(argv[1]);
    assert(NCores > 0);
    for (uint32_t coreId = 0; coreId < NCores; coreId++) {
        rv32i_5stage();
    }
}