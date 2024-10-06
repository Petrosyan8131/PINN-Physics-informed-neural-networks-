import sympy as sp
import scipy
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import numpy as np

gamma = 1.3
delta = 3
alpha = 0.001
beta = 0.0001
l = 1.25
omega = np.pi*l

dots = 500
x_0 = 0
x_T = 10.4
# tau = (x_T - x_0)/dots

# t = sp.symbols('t')
# x = sp.Function('x')

# equation = sp.Eq(x(t).diff(t, t) + delta*x(t).diff(t) + alpha*x(t) + beta*x(t), gamma*sp.cos(t))

# solution = sp.dsolve(equation)

# print(solution)

class Approx:
    def __init__(self, p = (gamma, delta, alpha, beta, omega)):
        self.p = p

    @staticmethod
    def F(t, s, gamma, delta, alpha, beta, omega):
        x, z = s
        dxdt = z
        dzdt = gamma*np.cos(omega*t) - delta*z - alpha*x - beta*x**3
        return dxdt, dzdt
    
    def solve(self):
        y0 = [1.0, 0.0]
        t_span = (x_0, x_T)
        t = np.linspace(x_0, x_T, dots)
        sol = solve_ivp(Approx.F, t_span, y0, args=self.p, method='RK45', t_eval=t)
        return sol

if __name__ == "__main__": # pragma: no cover
    t = np.linspace(x_0, x_T, dots)
    a = Approx()
    sol = a.solve()
    print(sol)
    plt.plot(t, sol.y[0, :], 'r')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    dl = (max(sol.y[0]) - min(sol.y[1]))*0.052 # <- change the last parameter to align the labels
    plt.text(t[-1]/2-2.8, max(sol.y[0, :])+1.3*dl, f"omega={'' if l==1 else l}pi")
    plt.text(t[-1]/2, max(sol.y[0])+1.6*dl, f"gamma={gamma}")
    plt.text(t[-1]/2, max(sol.y[0])+dl, f"delta={delta}")
    plt.text(t[-1]/2+2.5, max(sol.y[0])+1.6*dl, f"alpha={alpha}")
    plt.text(t[-1]/2+2.5, max(sol.y[0])+dl, f"beta={beta}")
    # plt.grid()
    plt.savefig(r"././figs/numerical_approx.png")
    plt.show()
