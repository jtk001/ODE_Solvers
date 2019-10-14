# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:11:31 2019

@author: Jonathan Kadowaki

This is an example usage of the Rk45 integrator solving the 2 body problem.

x'' + mu*x/(r^3) = 0
y'' + mu*y/(r^3) = 0
z'' + mu*z/(r^3) = 0

In the code, we have the following assignments:
    y1 = x
    y2 = x'
    y3 = y
    y4 = y'
    y5 = z
    y6 = z'
    
Initial conditions are given in km for position and km/h for velocities

"""

from RK45_v2 import RKF45
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# module level variables 
mu = (3.986004e14)*(3600**2)*(1000**(-3)) #grav. const in km^3h^-2

def f1(t,y):
    y2 = y[1] #dxdt
    dy1dt = y2
    return dy1dt

def f2(t,y):
    y1 = y[0] #x
    y3 = y[2] #y
    y5 = y[4] #z
    dy2dt = -(mu*y1)/((y1**2 + y3**2 + y5**2)**(3/2))
    return dy2dt

def f3(t,y):
    y4 = y[3] #dydt
    dy3dt = y4
    return dy3dt

def f4(t,y):
    y1 = y[0] #x
    y3 = y[2] #y
    y5 = y[4] #z
    dy4dt = -(mu*y3)/((y1**2 + y3**2 + y5**2)**(3/2))
    return dy4dt

def f5(t,y):
    y6 = y[5] #dydt
    dy5dt = y6
    return dy5dt

def f6(t,y):
    y1 = y[0] #x
    y3 = y[2] #y
    y5 = y[4] #z
    dy6dt = -(mu*y5)/((y1**2 + y3**2 + y5**2)**(3/2))
    return dy6dt

def main():
    #set up solution params
    period = 2.5 #hours per orbit
    n_orbits = 3 #how many orbits we want
    t0 = 0
    tinf = period*n_orbits #integration time
    y0 = np.array([-4777.8,
                   -6.7782*3600,
                   4862.6,
                   4.8929*3600,
                   1760.1,
                   0.9174*3600]) #y0 = [x0, x'0, y0, y'0, z0, z'0] in km and km/h
    
    #set up solver params
    step_seed = 0.1 #seed for step size
    e_upper = 1e-4
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
                      funcs=[f1,f2,f3,f4,f5,f6],
                      max_iter=max_iter)
    
    #get the solution and the associated error
    t,y,error = rk_solver.solve_ivp()
    
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    
    #plot the position 
    fig1 = plt.figure(num=1)
    ax1 = fig1.gca()
    ax1.plot(t,y[0],linewidth=2)
    ax1.plot(t,y[2],linewidth=2)
    ax1.plot(t,y[4],linewidth=2)
    ax1.set_xlabel('time (h)')
    ax1.set_ylabel('position (km)')
    ax1.legend(['x','y','z'])
    
    #plot the velocity
    fig2 = plt.figure(num=2)
    ax2 = fig2.gca()
    ax2.plot(t,y[1],linewidth=2)
    ax2.plot(t,y[3],linewidth=2)
    ax2.plot(t,y[5],linewidth=2)
    ax2.set_xlabel('time (h)')
    ax2.set_ylabel("velocity (km/h)")
    ax2.legend(['dx/dt','dy/dt','dz/dt'])
    
    #plot the parametric trajectory
    fig3 = plt.figure(num=3)
    ax3 = fig3.gca(projection='3d')
    ax3.plot(y[0],y[2],y[4],linewidth=2)
    ax3.set_xlabel('x (km)')
    ax3.set_ylabel('y (km)')
    ax3.set_zlabel('z (km')
    ax3.set_title('Two-Body Problem Trajectory')
    plt.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    main()