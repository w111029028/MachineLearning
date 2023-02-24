# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:35:58 2022

@author: A109021
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.integrate as integrate

distribution_function = lambda x: norm.pdf(x, 0.1)
x1 = 0
x2 = 1

print('probability to fall between {0} and {1} :'.format(x1, x2), integrate.quad(distribution_function, x1, x2)[0])
step = 0.001
whole_x = np.arange(-4, 4, step)
whole_y = list(map(distribution_function, whole_x))

needed_x = np.arange(x1, x2, step)
needed_y = list(map(distribution_function, needed_x))
plt.plot(whole_x, whole_y)
plt.fill_between(needed_x, needed_y)