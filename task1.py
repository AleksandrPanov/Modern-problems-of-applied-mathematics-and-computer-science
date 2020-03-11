import numpy
import matplotlib.pyplot as plt

def dichotomy(func, a, b, eps_x, eps_val):
    c = (a + b) / 2
    val = func(c)
    while b - a > eps_x and abs(val) > eps_val:
        if val < 0:
            a = c
        else:
            b = c
        c = (a + b) / 2
        val = func(c)
    return c

def newton(func, derivative, x, eps_val):
    val = func(x)
    while val > eps_val:
        x = x - func(x)/derivative(x)
    return x

class MyFunctor:
    def __init__(self, n, alpha):
        self.n = n
        self.alpha = alpha
    def __call__(self, x):
        return x**(self.n+1) + x - self.alpha

#calculate
eps_x = 0.001
eps_val = 0.001
x = numpy.arange(0, 2, 0.01)
alphas = numpy.arange(0, 100.1, 0.1)
ns = range(2, 8, 2)
roots = {2: [], 4: [], 6: []}
for n in ns:
    for alpha in alphas:
        myfunctor = MyFunctor(n, alpha)
        roots[n].append(dichotomy(myfunctor, 0.0, 10.0, eps_x, eps_val))

#output
fig, ax = plt.subplots(1, 4)
fig.set_size_inches(16, 6)
for n in ns:
    gr = ax[n//2 - 1]
    gr.plot(alphas, roots[n])
    gr.set_title('n = ' + str(n))
    gr.set_ylabel('root x')
    gr.set_xlabel('alpha')
    gr.grid()
gr = ax[3]
gr.set_title('n = ' + str(ns[-1]) + ', alpha = ' + str(alphas[-1]))
gr.set_ylabel('x^(n+1) + x - alpha')
gr.set_xlabel('x')
gr.plot(x, myfunctor(x))
fig.tight_layout()
plt.show()
