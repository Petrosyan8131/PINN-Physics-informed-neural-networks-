import numpy as np
import torch

class Runge_Kut:
    def __init__(self, dots=200, t_0=0.4, T=3.4, delta = 0.1, alpha = 1.2, beta = 2, gamma = 0.7, omega = torch.pi*1.75):
        self.dots = dots
        self.t_0 = t_0
        self.T = T
        self.tau = round((self.T - self.t_0)/self.dots, 3)
        self.t = np.arange(self.t_0, self.T+self.tau/2, self.tau)
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        
        # Начальное задание функции x(t)
        self.x_t = np.empty(self.dots+1, dtype=np.float64)
        self.z_t = np.empty(self.dots+1, dtype=np.float64)
        
    # Метод Рунге-Кутта
    def method(self):
        # построение
        for i in range(1, self.dots+1):
            k1x = self.z_t[i-1]
            k2x = self.z_t[i-1] + self.tau/2
            k3x = self.z_t[i-1] + self.tau/2
            k4x = self.z_t[i-1] + self.tau
            k1z = self.gamma*np.cos(self.omega*self.t[i]) - self.delta*self.z_t[i-1] - self.alpha*self.x_t[i-1] - self.beta*self.x_t[i-1]**3
            k2z = self.gamma*np.cos(self.omega*self.t[i]) - self.delta*(self.z_t[i-1] + self.tau/2*k1z) - self.alpha*(self.x_t[i-1]+self.tau/2) - self.beta*(self.x_t[i-1]+self.tau/2)**3
            k3z = self.gamma*np.cos(self.omega*self.t[i]) - self.delta*(self.z_t[i-1] + self.tau/2*k2z) - self.alpha*(self.x_t[i-1]+self.tau/2) - self.beta*(self.x_t[i-1]+self.tau/2)**3
            k4z = self.gamma*np.cos(self.omega*self.t[i]) - self.delta*(self.z_t[i-1] + self.tau*k3z) - self.alpha*(self.x_t[i-1]+self.tau) - self.beta*(self.x_t[i-1]+self.tau)**3
            self.x_t[i] = self.x_t[i-1] + self.tau/6*(k1x + 2*k2x + 2*k3x + k4x)
            self.z_t[i] = self.z_t[i-1] + self.tau/6*(k1z + 2*k2z + 2*k3z + k4z)
        return self.x_t, self.z_t, self.t, self.t_0, self.T

if __name__ == "__main__": # pragma: no cover
    a = Runge_Kut()
    a.method()