import numpy
import matplotlib.pyplot as plt
from numba import njit
from math import log, exp, sqrt, cos, sin
from numpy.linalg import norm

x0 = numpy.array([1.])
x0_2 = numpy.array([1., 0.])
x0_3 = numpy.array([1., 1.0, 0.])
hs = [0.1, 0.01, 0.001]

@njit
def func1_1(t, xi):
    return xi
    
@njit
def func1_1_sol(t, x0):
    return exp(t)*x0

@njit
def func1_2(t, xi):
    return -xi

@njit
def func1_2_sol(t, x0):
    return exp(-t)*x0

@njit
def func2(t, x0):
    return numpy.array([x0[1], -x0[0]])
    
@njit
def func3(t, tmp):
    x = tmp[0]
    y = tmp[1]
    z = tmp[2]
    return numpy.array([-y-z, x+0.2*y, 0.2+(x-5.7)*z])

@njit
def func2_sol(t, x0):
    return numpy.array([cos(t), -sin(t)])

@njit
def methodEuler(x, f, h):
    n = len(x)
    for i in range(n-1):
        t = i*h
        x[i+1] = x[i] + h * f(t, x[i])
    return x

@njit
def methodRungeKutta(x, f, h):
    n = len(x)
    for i in range(n-1):
        t = i*h
        k1 = f(t, x[i])
        k2 = f(t + 0.5*h, x[i] + 0.5*h*k1)
        k3 = f(t + 0.5*h, x[i] + 0.5*h*k2)
        k4 = f(t + h, x[i] + h*k3)
        x[i+1] = x[i] + h*(k1 + 2*k2 + 2*k3 + k4)/6.
    return x

def getError(xs, f, ts, x0):
    return [abs(xs[i]-f(ts[i], x0)) for i in range(len(xs))]

@njit
def getError2(xs, f, ts, x0):
    return [norm(xs[i]-f(ts[i], x0)) for i in range(len(xs))]
    
@njit
def getError3(xs1, xs2):
    return [norm(xs1[i]-xs2[2*i]) for i in range(len(xs1))]

# output task
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(22, 12)
res_fig, res_gr = plt.subplots(3)
res_fig.set_size_inches(11, 12)

gr1, gr2 = ax[0][0], ax[1][0]
gr3, gr4 = ax[0][1], ax[1][1]

def initCond(h, T, x0):
    ts = numpy.arange(0, T, h)
    n = int(0.5 + T / h)
    xs = numpy.zeros(shape=(n, len(x0)), dtype=numpy.float64)
    xs[0] = x0
    return xs, ts

i = 0
for h in hs:   
    # task 1
    xs, ts = initCond(h, 20.0, x0)
    methodEuler(xs, func1_1, h)
    gr1.plot(ts, getError(xs, func1_1_sol, ts, x0), label="h="+str(h))
    if h == hs[-1]:
        res_gr[0].plot(ts, getError(xs, func1_1_sol, ts, x0), label='метод Эйлера')
    
    methodRungeKutta(xs, func1_1, h)
    gr3.plot(ts, getError(xs, func1_1_sol, ts, x0), label="h="+str(h))
    if h == hs[-1]:
        res_gr[0].plot(ts, getError(xs, func1_1_sol, ts, x0), label='метод Рунге-Кутта')
    
    methodEuler(xs, func1_2, h)
    gr2.plot(ts, getError(xs, func1_2_sol, ts, x0), label="h="+str(h))
    
    methodRungeKutta(xs, func1_2, h)
    gr4.plot(ts, getError(xs, func1_2_sol, ts, x0), label="h="+str(h))
    
    # task 2
    xs, ts = initCond(h, 100.0, x0_2)
    methodEuler(xs, func2, h)
    res_gr[1].plot(ts, getError2(xs, func2_sol, ts, x0_2), label="h="+str(h))
    #methodRungeKutta(xs, func2, h)
    #res_gr[1].plot(ts, getError2(xs, func2_sol, ts, x0_2), label="h="+str(h))
    
    # task 3
    xs1, ts1 = initCond(h, 1000.0, x0_3)
    methodRungeKutta(xs1, func3, h)
    #
    xs2, ts2 = initCond(h/2., 1000.0, x0_3)
    methodRungeKutta(xs2, func3, h/2.)
    res_gr[2].plot(ts1, getError3(xs1, xs2), label="h="+str(h))  
    i+=1

plt.rcParams.update({'font.size': 12})

def initGraph(gr, name):
    gr.set_yscale('log')
    gr.set_title(name, fontsize=16)
    gr.set_xlabel('t', fontsize=14)
    gr.set_ylabel('error', fontsize=14)
    gr.legend(loc='best')
    gr.grid()

initGraph(gr1, "Ошибка метода Эйлера, x'=x")
initGraph(gr2, "Ошибка метода Эйлера, x'=-x")
initGraph(gr3, "Ошибка метод Рунге-Кутта 4-го порядка, x'=-x")
initGraph(gr4, "Ошибка метод Рунге-Кутта 4-го порядка, x'=-x")
initGraph(res_gr[0], "Ошибка методов Эйлера и Рунге-Кутта, x'=x")
initGraph(res_gr[1], "Ошибка мeтода Эйлера, x'' + x = 0")
initGraph(res_gr[2], "Ошибка мeтода Рунге-Кутта, система Ресслера")


fig.tight_layout()
res_fig.tight_layout()
plt.show()
