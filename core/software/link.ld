MEMORY
{
    IMEM : ORIGIN = 0x00000000, LENGTH = 4K
    DMEM : ORIGIN = 0x80000000, LENGTH = 8K
}

SECTIONS
{
    .text :
    {
        *(.text)
        *(.text.*)
    } > IMEM

    .data :
    {
        *(.data)
        *(.data.*)
    } > DMEM
}