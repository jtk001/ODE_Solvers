# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:51:35 2019

@author: Jonathan Kadowaki

This is a pure python implementation of the RKF45 method.

"""

import numpy as np

class RKF45():
    '''
    This class will solve systems of first order ODEs via RKF45 method. The
    step size will be adjusted s.t. the error will remain between an upper and a
    lower bound. 
    
    upper_tol - float, upper tolerance for error
    lower_tol - float, lower tolerance for error
    t0 - float, initial time to start solution
    tinf - float, stop time
    y0 - np array containing the initial conditions np.array([y1(0), ... yn(0)])
    funcs - list of RHS functions, eg [f1(t,y1,y2), f2(t,y1,y2)]
    max_iter - max number of inner iterations for step size adaptation
    
    Usage:
        1. create rhs functions eg y1'=f1(t,y1,y2), y2'=f2(t,y1,y2) as functions of t and y
        where y = [y1,y2]
        2. create initial conditions y0 = [y1(0), y2(0)]
        3. decide on error tolerances and starting step size
        4. instantiate the solver object
        5. call the self.solve_ivp() public method to get the solution
        
    '''
    def __init__(self,upper_tol,
                 lower_tol,
                 step_size,
                 t0,
                 tinf,
                 y0,
                 funcs,
                 max_iter=100):
        self.e_u = upper_tol
        self.e_l = lower_tol
        self.step_size = step_size
        if type(y0) != 'numpy.ndarray':
            self.y0 = np.array(y0)
        else:
            self.y0 = y0
        self.t0 = t0
        self.tinf = tinf
        self.max_iter = max_iter
        
        # construct the coefficients from Butcher Table
        self.a = np.array([[0,0,0,0,0],
                           [1/4,0,0,0,0],
                           [3/32,9/32,0,0,0],
                           [1932/2197,-7200/2197,7296/2197,0,0],
                           [439/216,-8,3680/513,-845/4104,0],
                           [-8/27,2,-3544/2565,1859/4104,-11/40]])
        self.c = np.array([0,1/4,3/8,12/13,1,1/2])
        self.b_4 = np.array([25/216,0,1408/2565,2197/4104,-1/5,0])
        self.b_5 = np.array([16/135,0,6656/12825,28561/56430,-9/50,2/55])
        self.b = np.vstack((self.b_4,self.b_5))
        
        #hold solutions in a list
        self.y = [y0]
        self.times = [t0]
        self.error = [0]
        self.p = 6
        self.funcs = funcs
        
    def _rhs_eval(self,t,y):
        rhs_eval = []
        for func in self.funcs:
            rhs_eval.append(func(t,y))
        return np.array(rhs_eval) #(N,)
            
    def _compute_k1(self,t,y):
        return self._rhs_eval(t,y)
    
    def _compute_k2(self,t,y,h,k1):
        t_next = t + self.c[1]*h
        y_next = y + h*self.a[1,0]*k1
        return self._rhs_eval(t_next,y_next)
    
    def _compute_k3(self,t,y,h,k1,k2):
        k = np.vstack((k1,k2)) #(2,N)
        a = self.a[2,0:2,np.newaxis] #(2,1)
        t_next = t + self.c[2]*h
        y_next = y + h*np.sum(a*k,axis=0)
        return self._rhs_eval(t_next,y_next)
    
    def _compute_k4(self,t,y,h,k1,k2,k3):
        k = np.vstack((k1,k2,k3)) #(3,N)
        a = self.a[3,0:3,np.newaxis] #(3,1)
        t_next = t + self.c[3]*h
        y_next = y + h*np.sum(a*k,axis=0)
        return self._rhs_eval(t_next,y_next)
    
    def _compute_k5(self,t,y,h,k1,k2,k3,k4):
        k = np.vstack((k1,k2,k3,k4)) #(4,N)
        a = self.a[4,0:4,np.newaxis] #(4,1)
        t_next = t + self.c[4]*h
        y_next = y + h*np.sum(a*k,axis=0)
        return self._rhs_eval(t_next,y_next)
    
    def _compute_k6(self,t,y,h,k1,k2,k3,k4,k5):
        k = np.vstack((k1,k2,k3,k4,k5)) #(5,N)
        a = self.a[5,0:5,np.newaxis] #(5,1)
        t_next = t + self.c[5]*h
        y_next = y + h*np.sum(a*k,axis=0)
        return self._rhs_eval(t_next,y_next)
    
    def _compute_k(self,t,y,h):
        k1 = self._compute_k1(t,y)
        k2 = self._compute_k2(t,y,h,k1)
        k3 = self._compute_k3(t,y,h,k1,k2)
        k4 = self._compute_k4(t,y,h,k1,k2,k3)
        k5 = self._compute_k5(t,y,h,k1,k2,k3,k4)
        k6 = self._compute_k6(t,y,h,k1,k2,k3,k4,k5)
        k = np.vstack([k1,k2,k3,k4,k5,k6])
        return k #(6,N) array
    
    def _compute_next_step(self,t,y,h):
        k = self._compute_k(t,y,h) #(6,N) array, N = num of equations
        y_lower = y + h*np.sum(self.b[0,:,np.newaxis]*k,axis=0) #lower order method step
        t_next = t + h
        error = self._compute_error(k,h) #grab error against higher order method
        return y_lower,t_next,error
    
    def _compute_error(self,k,h):
        b = np.diff(self.b,axis=0).reshape((-1,1)) # (bi* - bi), reshaped to (6,1)
        error_vec = h*np.sum(k*b,axis=0)
        error = np.linalg.norm(error_vec)
        return error
    
    def _update_step_size(self,error,h):
        if error < self.e_l:
            violated_bound = self.e_l
        else:
            violated_bound = self.e_u    
        h_new = np.sqrt(violated_bound/np.abs(error))*h
        return h_new
    
    def _post_process_soln(self):
        y = np.array(self.y).T
        return y
    
    def solve_ivp(self):
        '''
        This is the only public method for this class. Call this after 
        instantiation to solve the ivp.
        '''
        h = self.step_size
        t = self.t0
        y = self.y0
        print('Begin RKF45, t = {}'.format(str(t)))
        while t < self.tinf:
            y_next,t_next,err = self._compute_next_step(t,y,h)
            j = 0
            while not self.e_l < err < self.e_u:
                h = self._update_step_size(err,h)
                y_next,t_next,err = self._compute_next_step(t,y,h)
                j+=1
                if j > self.max_iter:
                    print('time step not converged, advancing to t = {} anyway'.format(str(t_next)))
                    break
            print('Solution advanced to t = {}'.format(str(t_next)))
            self.y.append(y_next)
            self.times.append(t_next)
            self.error.append(err)
            y = y_next
            t = t_next
           
        y_final = self._post_process_soln()
        print('RKF45 finished, t = {}'.format(str(self.times[-1])))
        return (self.times,y_final,self.error)
        
