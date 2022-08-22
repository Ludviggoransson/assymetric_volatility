"""
Evaluating the derivation of Standard Classical Laplace (SCL) distribution to a difference of two exponential.
Model 1.3.1: Modeling volatility process, with standard exponential.
"""
import numpy as np
from typing import Optional
import plotly.express as px


class VolatilityProcess():
    """
    Class to generate and evaluate assymmetric volatility. 
    """
    def __init__(self, alpha:float=0.19, beta:float=0.2, theta:float=0.0,shape:int=1):
        """
        Parameters for the Standard Laplace process that fulfills the restraint of 2\alpha < 1-\beta with zero shocks. 

        Args:
            alpha (float, optional): Defaults to 0.19.
            beta (float, optional): Defaults to 0.3.
            theta (float, optional): Defaults to 0.0 as in zero shocks. 
            shape (int, optional): Defaults to 1. Shaper parameter of the exponential processes. 
        """
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.shape = shape
    
    def exponential_process(self, n:int)->np.array:
        """
        Creates stochastic volatiltiy process of length n. 

        Args:
            n (int): Length of the volatility process. 

        Returns:
            np.array: numpy array of the volatility process of length n. 
        """
        exp1 = np.random.exponential(scale=self.shape, size=n)
        exp2 = np.random.exponential(scale=self.shape, size=n)
        rho = np.full(n, np.nan)
        rho[0] = 1

        for i in range(1, len(rho)):
            rho[i] = 1+rho[i-1]*(
                                self.alpha*
                                ((1-self.theta)*exp1[i-1]+(1+self.theta)*exp2[i-1])+self.beta
                                )
        return rho
