import numpy
import matplotlib.pyplot as plt


def dichotomy(func, a, b, eps_x, eps_val, max_it=1000000):
    c = (a + b) / 2
    val = func(c)
    it = 1
    while b - a > eps_x and abs(val) > eps_val and it < max_it:
        if val < 0:
            a = c
        else:
            b = c
        c = (a + b) / 2
        val = func(c)
        it += 1
    return c


def newton(func, derivative, x, eps_val, max_it=1000000):
    val = func(x)
    it = 0
    while abs(val) > eps_val and it < max_it:
        x = x - func(x)/derivative(x)
        it += 1
        val = func(x)
    return x


class ProteinSynthesisFunc:
    def __init__(self, n, alpha):
        self.n = n
        self.alpha = alpha

    def __call__(self, x):
        return x**(self.n+1) + x - self.alpha


class DerivativeProteinSynthesisFunc:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return (self.n + 1) * x**self.n + 1

# calculate task 1.2
eps_x = 0.0001
eps_val = 0.0001
dih_b = 10.0
dih_a = -5.0
x = numpy.arange(0, 2, 0.01)
alphas = numpy.arange(0.001, 80.0, 0.01)
ns = range(2, 8, 2)
roots = {2: [], 4: [], 6: []}
for n in ns:
    myDerivative = DerivativeProteinSynthesisFunc(n)
    for alpha in alphas:
        myFunctor = ProteinSynthesisFunc(n, alpha)
        x = newton(myFunctor, myDerivative, 10.0, 0.00001)

        L = -n*(x**(n+1))/alpha
        if abs(L) < 1:
            continue
        absBetta = numpy.sqrt(L*L - 1)
        T = abs(numpy.arccos(1/L)) / absBetta
        roots[n].append(T)
# output task
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
gr = ax
gr.set_xscale("log")
for n in ns:
    gr.plot(roots[n], alphas[len(alphas) - len(roots[n]):], label='n = ' + str(n))

plt.rcParams.update({'font.size': 18})
plt.xlabel('xlabel', fontsize=16)
plt.ylabel('ylabel', fontsize=16)

gr.legend(loc='upper right')
gr.set_title('Alpha от $\\tau$')
gr.set_xlabel('$\\tau$')
gr.set_ylabel('Alpha')
gr.grid()


fig.tight_layout()
plt.show()
