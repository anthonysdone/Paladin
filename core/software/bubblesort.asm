.section .text
.global _start

_start: 
    li x3, 16          # x3 = n
    li x4, 0           # x4 = i

outer_loop:
    bge x4, x3, done    # if i >= n, done
    li x5, 0            # x5 = j

inner_loop: 
    addi x6, x3, -1         # x6 = n - 1
    sub x6, x6, x4          # x6 = n - 1 - i
    bge x5, x6, outer_inc   # if j >= n - 1 - i, next outer loop

    slli x7, x5, 2          # x7 = j * 4
    lw x8, 0(x7)            # x8 = arr[j]
    lw x9, 4(x7)            # x9 = arr[j + 1]

    ble x8, x9, no_swap     # if arr[j] <= arr[j + 1], dont swap

    # else, swap
    sw x9, 0(x7)            # arr[j] = arr[j + 1]
    sw x8, 4(x7)            # arr[j + 1] = arr[j]

no_swap:
    addi x5, x5, 1          # j++
    jal x0, inner_loop

outer_inc:
    addi x4, x4, 1          # i++
    jal x0, outer_loop

done: 
    jal x0, done            # stop
