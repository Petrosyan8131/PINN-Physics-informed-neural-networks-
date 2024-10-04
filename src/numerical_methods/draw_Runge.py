import matplotlib.pyplot as plt
import numpy as np
from Runge_kut_approx import Runge_Kut

a = Runge_Kut()
t = a.method()

# Создание окна
ax = plt.gca()
plt.plot(t[2], t[0], label=r"function")
plt.plot(t[2], t[1], label=r"first derivative")

plt.xlim(t[3], t[4])
plt.ylim(-1, 1)
plt.legend(fontsize=12)
plt.title("Numerical study")
plt.tight_layout()
plt.savefig(r"figs/runge_kut.png")
plt.show()
