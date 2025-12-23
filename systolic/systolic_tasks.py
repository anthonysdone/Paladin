import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import *

@task
def mac(a_in, b_in, sum, a_out, b_out):
    sum.next = sum.val + a_in.val * b_in.val
    a_out.next = a_in.val
    b_out.next = b_in.val

@task
def feed_a_row(cycle, row_idx, matrix_a, a_out, N):
    if cycle.val >= row_idx and cycle.val < row_idx + N:
        col = cycle.val - row_idx
        a_out.next = matrix_a[row_idx][col]
    else:
        a_out.next = 0

@task
def feed_b_col(cycle, col_idx, matrix_b, b_out, N):
    if cycle.val >= col_idx and cycle.val < col_idx + N:
        row = cycle.val - col_idx
        b_out.next = matrix_b[row][col_idx]
    else:
        b_out.next = 0

@task
def counter(cycle):
    cycle.next = cycle.val + 1