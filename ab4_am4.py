# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:15:20 2019

@author: Jonathan Kadowaki

"""

import numpy as np

class AB4AM4():
    def __init__(self,
                 step_size,
                 funcs,
                 t0,
                 tinf,
                 y0,
                 tol,
                 max_iter=1000):
        
        self.step_size = step_size
        self.funcs = funcs
        self.max_iter = max_iter
        self.t0 = t0
        self.tinf = tinf
        self.y0 = y0
        self.y = [y0]
        self.times = [t0]
        self.history = []
        self.next_f = None
        self.tol = tol
        
        self.ab4_coeff = np.array([55,-59,-37,-9])
        self.am4_coeff = np.array([9,19,-5,1])
        
    def _rhs_eval(self,t,y):
        rhs_eval = []
        for func in self.funcs:
            rhs_eval.append(func(t,y))
        return np.array(rhs_eval)
        
    def _rk4_step(self,t,y):
        t1,y1 = t,y
        h = self.step_size
        k1 = h*self._rhs_eval(t1,y1)
        t2 = t1 + (h/2)
        y2 = y1 + (k1/2)
        k2 = h*self._rhs_eval(t2,y2)
        t3 = t2
        y3 = y1 + (k2/2)
        k3 = h*self._rhs_eval(t3,y3)
        t4 = t1 + h
        y4 = y1 + k3
        k4 = h*self._rhs_eval(t4,y4) 
        y_next = y1 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        return y_next
    
    def _kickstart(self):
        h = self.step_size
        t = self.t0
        y = self.y0
        self.history.append(self._rhs_eval(t,y)) #f(t0,y0)
        for _ in range(2): #perform 2 rk4 steps to build solution history -> will have 3 past f evaluations
            t_next = t + h
            y_next = self._rk4_step(t,y)
            self.y.append(y_next)
            self.times.append(t_next)
            self.history.append(self._rhs_eval(t_next,y_next))
            t = t_next
            y = y_next
        return (t,y)
    
    def _check_error(self,y_true,y_pred):
        error = y_true - y_pred
        error_norm = np.linalg.norm(error)
        if error_norm > self.tol:
            return False
        else:
            return True
        
    def _ab4_step(self,t,y):
        h = self.step_size
        f_curr = self._rhs_eval(t,y) # evaluate rhs at current solution point
        self.history.append(f_curr)
        int_approx = self.ab4_coeff[0]*f_curr
        for k,coeff in enumerate(self.ab4_coeff[1:]):
            int_approx += coeff*self.history[-1-(k+1)]
        y_next = y + (h/24)*int_approx
        return y_next

    def _am4_step(self,t,y,y_pred):
        h = self.step_size
        t_next = t + h
        f_approx = self._rhs_eval(t_next,y_pred)
        int_approx = self.am4_coeff[0]*f_approx
        for k,coeff in enumerate(self.am4_coeff[1:]):
            int_approx += coeff*self.history[-1-k]
        y_next = y + (h/24)*int_approx
        return y_next
    
    def _post_process(self):
        self.y = np.array(self.y).T
    
    def solve_ivp(self):
        h = self.step_size
        t,y = self._kickstart() #build history with 2 rk4 steps
        while t < self.tinf:
            t_next = t + h
            y_pred = self._ab4_step(t,y) #predictor for t_next
            y_corr = self._am4_step(t,y,y_pred)
            satisfied = self._check_error(y_corr,y_pred)
            i = 0
            while not satisfied:
                y_pred = y_corr
                y_corr = self._am4_step(t,y,y_pred)
                satisfied = self._check_error(y_corr,y_pred)
                i += 1
                if i > self.max_iter:
                    print('Error bound not satisfied, advancing to {} anyway'.format(t_next))
                    break
                
            self.y.append(y_corr)
            self.times.append(t_next)
            y = y_corr
            t = t_next
            
        self._post_process()
        
        return (self.times,self.y)