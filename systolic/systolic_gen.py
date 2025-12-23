import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import *
from systolic_tasks import *

def gen_systolic(matrix_a, matrix_b):
    N = len(matrix_a)
    sim = Sim()

    count = sim.reg(0)
    sim.add(counter(count))

    a_feeders = [[sim.reg(0) for _ in range(N)] for _ in range(N)]
    b_feeders = [[sim.reg(0) for _ in range(N)] for _ in range(N)]

    mac_accums = [[sim.reg(0) for _ in range(N)] for _ in range(N)]

    a_in_feeders = [sim.reg(0) for _ in range(N)]
    b_in_feeders = [sim.reg(0) for _ in range(N)]

    for i in range(N):
        sim.add(feed_a_row(count, i, matrix_a, a_in_feeders[i], N))
        sim.add(feed_b_col(count, i, matrix_b, b_in_feeders[i], N))
    
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
                mac_accums[i][j],
                get_a_out(i, j),
                get_b_out(i, j)
            ))
    
    return sim, mac_accums

if __name__ == "__main__":
    N = 4

    matrix_a = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    matrix_b = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    sim, mac_accums = gen_systolic(matrix_a, matrix_b)

    for i in range(3 * N):
        sim.run(1)
        print(f"\nCycle {i}:")
        for i in range(N):
            row = []
            for j in range(N):
                row.append(mac_accums[i][j].val)
            print(row)

    # sim.run(3 * N)

    # expected = [[0 for _ in range(N)] for _ in range(N)]
    # for i in range(N):
    #     for j in range(N):
    #         for k in range(N):
    #             expected[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    # print("Actual:")
    # for row in expected:
    #     print(row)
    
    # print("\nSystolic:")
    # for i in range(N):
    #     row = []
    #     for j in range(N):
    #         row.append(mac_accums[i][j].val)
    #     print(row)