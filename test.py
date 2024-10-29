import torch
import torch.nn as nn

# from Activation_functions.Activation_cos import Cos
# from numerical_methods.Runge_kut_scipy import Approx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# from numerical_methods.Runge_kut_approx import Runge_Kut

from torch.utils.data import Dataset, DataLoader
# from Activation_functions.Activation_sin_cos import Sin, Cos

a = torch.tensor([1.009, 1.007, 1.004])

b = torch.tensor([1.007, 1.004, 1.002]).unsqueeze(1)

print(torch.max(torch.abs(a-b)))