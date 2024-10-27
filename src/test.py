import torch
import torch.nn as nn

# from Activation_functions.Activation_cos import Cos
from numerical_methods.Runge_kut_scipy import Approx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from numerical_methods.Runge_kut_approx import Runge_Kut

from torch.utils.data import Dataset, DataLoader
from Activation_functions.Activation_sin_cos import Sin, Cos

from tqdm import tqdm
# import optuna

# Используем доступные графические процессоры
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dots = 500

# Задаем начальные данные для апроксимации (задание сетки точек)
t = (torch.linspace(0, 10.4, dots).unsqueeze(1)).to(device)
t.requires_grad = True
t_in = t[1:]
t_bc = t[0]

f_true = (torch.zeros(dots-1).unsqueeze(1)).to(device)
f_true.requires_grad = True

print(f_true)
