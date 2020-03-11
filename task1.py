import numpy
import matplotlib.pyplot as plt

def dichotomy(func, a, b, eps_x, eps_val):
    c = (a + b) / 2
    val = func(c)
    while b - a > eps_x and val > eps_val:
        if val < 0:
            a = c
        else:
            b = c
        c = (a + b) / 2
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

#test output
x = numpy.arange(0, 2, 0.01)
myfunctor = MyFunctor(2, 1.0)
y = myfunctor(x)
fig, ax = plt.subplots(1, 2)
gr1 = ax[0]
gr2 = ax[1]
gr1.plot(x, y)
for gr in ax:
    gr.set_ylabel('y')
    gr.set_xlabel('x')
    gr.grid()
fig.tight_layout()
plt.show()
