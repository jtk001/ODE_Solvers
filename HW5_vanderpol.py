# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:11:31 2019

@author: Jonathan Kadowaki

This is an example usage of Rkf45 integrator, solving a forced van der pol
oscillator problem.

y'' + y - mu*(1-y^2)*y' = Asin(wt)

In the code we have the following assignments:
    y1 = y
    y2 = y'

"""

from RK45_v2 import RKF45
import numpy as np
import matplotlib.pyplot as plt

def f1(t,y):
    y2 = y[1]
    return y2

def f2(t,y):
    y1 = y[0]
    y2 = y[1]
    A = 1
    mu = 1/2
    w = 10
    rhs = A*np.sin(w*t) + mu*(1-y1**2)*y2 - y1
    return rhs

def main():
    #set up solution params
    t0 = 0
    tinf = 100
    y0 = np.array([1,1])
    
    #set up solver params
    step_seed = 0.1 #seed for step size
    e_upper = 1e-9
    e_factor = 1e-3
    e_lower = e_upper*e_factor
    max_iter = 100000 #max iter for step size adaptations
    
    #create rk solver object
    rk_solver = RKF45(upper_tol=e_upper,
                      lower_tol=e_lower,
                      step_size=step_seed,
                      t0=t0,
                      tinf=tinf,
                      y0=y0,
                      funcs=[f1,f2],
                      max_iter=max_iter)
    
    #get the solution
    t,y,error = rk_solver.solve_ivp()
    
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    
    fig1 = plt.figure(num=1)
    plt.plot(t,y[0],linewidth=2)
    plt.xlabel('time')
    plt.ylabel('y(t)')
    
    fig2 = plt.figure(num=2)
    plt.plot(t,y[1],linewidth=2)
    plt.xlabel('time')
    plt.ylabel("y'(t)")
    
    fig3 = plt.figure(num=3)
    plt.plot(y[0],y[1],linewidth=1)
    plt.xlabel('y(t)')
    plt.ylabel("y'(t)")
    
    fig4 = plt.figure(num=4)
    plt.plot(t,error,linewidth=1)
    plt.xlabel('time')
    plt.ylabel("relative error")
    
    plt.show()

if __name__ == '__main__':
    main()