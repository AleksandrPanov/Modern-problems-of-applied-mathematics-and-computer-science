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

class MyFunctor:
    def __init__(self, n, alpha):
        self.n = n
        self.alpha = alpha
    def __call__(self, x):
        return x**(self.n+1) + x - self.alpha

#test output
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some y')
plt.xlabel('some x')
plt.show()
myfunctor = MyFunctor(2, 1.0)
print(myfunctor(0.5))
