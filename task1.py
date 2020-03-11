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
eps_x = 0.001
eps_val = 0.001
dih_b = 10.0
dih_a = -5.0
x = numpy.arange(0, 2, 0.01)
alphas = numpy.arange(0, 80.1, 0.1)
ns = range(2, 8, 2)
roots = {2: [], 4: [], 6: []}
for n in ns:
    for alpha in alphas:
        myFunctor = ProteinSynthesisFunc(n, alpha)
        roots[n].append(dichotomy(myFunctor, dih_a, dih_b, eps_x, eps_val))

# output task 1.2
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(16, 6)
gr = ax[0]
for n in ns:
    gr.plot(alphas, roots[n], label='n = ' + str(n))
gr.legend(loc='upper left')
gr.set_title('Корни f(x) при различных n и alpha')
gr.set_xlabel('Alpha')
gr.set_ylabel('Значение корня x*')
gr.grid()

gr = ax[1]
gr.grid()
gr.set_title('График f(x) при различных n, alpha = 5.0')
gr.set_xlabel('x')
gr.set_ylabel('f(x)')
myFunctor = ProteinSynthesisFunc(2, 5.0)
gr.plot(x, myFunctor(x), label='n = 2')
myFunctor.n = 4
gr.plot(x, myFunctor(x), label='n = 4')
myFunctor.n = 6
gr.plot(x, myFunctor(x), label='n = 6')
gr.legend(loc='upper left')

fig.tight_layout()
plt.show()

# calc task 1.3
myFunctor = ProteinSynthesisFunc(2, 0.0)
myDerivative = DerivativeProteinSynthesisFunc(2)
count = range(1, 51, 1)
dihError = []
newtonError = []
newtonErrorX = []
dihErrorXa = []
dihErrorXb = []

dih_interval = dih_b - dih_a
for c in count:
    x1 = dichotomy(myFunctor, dih_a, dih_b, 0, 0, c)
    dihError.append(abs(myFunctor(x1)))
    x2 = newton(myFunctor, myDerivative, 100.0, 0.0, c)
    newtonError.append(abs(myFunctor(x2)))
    newtonErrorX.append(x2)
    dihErrorXa.append(x1 - dih_interval / 2)
    dihErrorXb.append(x1 + dih_interval / 2)
    dih_interval /= 2.0

# output task 1.3
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(16, 6)
for gr in ax:
    gr.set_xlabel('Число итераций')
    gr.grid()
ax[0].set_yscale('log')
ax[0].set_ylabel('abs(f($x_n$))')
ax[0].set_title('Значение abs(f(x)) в точке x*. Оси в логарифмическом масштабе')
ax[0].plot(count, dihError, label='метод дихотомии')
ax[0].plot(count, newtonError, label='метод Ньютона')
ax[0].legend(loc='upper right')
ax[1].set_ylabel('$x_n$')
ax[1].set_title('Метод Ньютона, значение x*')
ax[1].plot(count, newtonErrorX)

ax[2].set_ylabel('$x_n$')
ax[2].set_title('Метод дихотомии, значение x*')
ax[2].plot(count, dihErrorXa, label='Значение a')
ax[2].plot(count, dihErrorXb, label='Значение b')
ax[2].legend(loc='upper right')

fig.tight_layout()
plt.show()