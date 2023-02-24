# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:42:57 2022

@author: A109021
"""

import math
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+math.exp(0.1*-x + 20))


def plot(px, py):
    plt.plot(px, py)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    plt.show()


def main():
    # Init
    x = []
    dx = -2000
    while dx <= 2000:
        x.append(dx)
        dx += 0.1

    # Use sigmoid() function
    px = [xv for xv in x]
    py = [sigmoid(xv) for xv in x]

    # Plot
    plot(px, py)


if __name__ == "__main__":
    main()

