import numpy
import matplotlib.pyplot as plt
from numba import njit
from numba import jit
from math import log

x_0 = 0.5
start = 10000
num_y = 1000
start_r = 0.
max_r = 4.
delta_r = 0.001
num_r = int((max_r - start_r) / delta_r) + 1
xs = numpy.zeros(shape=(num_y), dtype=numpy.float64)

@njit(cache=True)
def calculate(x, start, r):
    for i in range(start):
        x = x*(r-x)
    return x

@njit(cache=True)
def calculate_res(xs, num_y, r):
    for i in range(1, num_y):
        xs[i] = xs[i-1]*(r-xs[i-1])

def drawGr(xs, x_0, start, num_y, num_r, delta_r, start_r, gr1):
    for step in range(0, num_r):
        r = step * delta_r + start_r
        x_start = calculate(x_0, start, r)
        rs = numpy.full(num_y, r)
        xs[0] = x_start
        calculate_res(xs, num_y, r)
        gr1.plot(rs, xs, 'g.', markersize=0.05)
        gr2.plot(r, getLyapunov(xs, r, num_y), 'bo', markersize=0.5)

@njit(cache=True)
def getLyapunov(xs, r, num_y):
    tmp = 0.0
    for x in xs:
        tmp += log(abs(r-2*x))
    return tmp/num_y

print(calculate(x_0, 2000, 4.))        
# при r > 4 значение x стремится к минус бесконечности
print(calculate(x_0, 2000, 4.001))
print(calculate(x_0, 2000, 4.01))

# output task
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(22, 8)
gr1, gr2 = ax[0], ax[1]

drawGr(xs, x_0, start, num_y, num_r, delta_r, start_r, gr1)

plt.rcParams.update({'font.size': 18})
plt.xlabel('xlabel', fontsize=16)
plt.ylabel('ylabel', fontsize=16)

gr1.set_title('Диаграмма ветвления')
gr1.set_xlabel('r')
gr1.set_ylabel('x')
gr1.grid()

gr2.set_title('Значение ляпуновского показателя')
gr2.set_xlabel('r')
gr2.set_ylabel('$\\lambda$')
gr2.grid()


fig.tight_layout()
plt.show()
