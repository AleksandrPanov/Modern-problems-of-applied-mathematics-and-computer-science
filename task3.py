import numpy
import matplotlib.pyplot as plt
from numba import njit
from numba import jit
x_0 = 0.5
start = 10000
num_y = 1000
start_r = 0.
max_r = 4.
delta_r = 0.001
num_r = int((max_r - start_r) / delta_r) + 1
x = numpy.zeros(shape=(num_y), dtype=numpy.float64)

@njit(cache=True)
def calculate(x, start, r):
    for i in range(start):
        x = x*(r-x)
    return x

@njit(cache=True)
def calculate_res(x, num_y, r):
    for i in range(1, num_y):
        x[i] = x[i-1]*(r-x[i-1])


def drawGr(x, x_0, start, num_y, num_r, delta_r, start_r, gr):
    for step in range(0, num_r):
        r = step * delta_r + start_r
        x_start = calculate(x_0, start, r)
        rs = numpy.full(num_y, r)
        x[0] = x_start
        calculate_res(x, num_y, r)
        gr.plot(rs, x, 'g.', markersize=0.2)

# output task
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
gr = ax

drawGr(x, x_0, start, num_y, num_r, delta_r, start_r, gr)

plt.rcParams.update({'font.size': 18})
plt.xlabel('xlabel', fontsize=16)
plt.ylabel('ylabel', fontsize=16)

gr.set_title('Диаграмма ветвления')
gr.set_xlabel('r')
gr.set_ylabel('x')
gr.grid()


fig.tight_layout()
plt.show()
