"""
Evaluation of Maximum Likelihood estimators of alpha, beta and theta.  
"""
import numpy as np
from pdf_decomposed_standard_laplace import DecomposedSLC

class MLEstimator(DecomposedSLC):
    """
    Class to generate Maximum Likehood estimators from decomposing a Standard Classical Laplace to a difference of two exponential processes. 
    """
    def __init__(self, alpha:float, beta:float, theta:float, rho:np.array):
        """Input parameters for the ML estimators. 

        Args:
            alpha (float): alpha parameter. 
            beta (float): beta parameter.
            theta (float): theta parameter.
            rho (np.array): volatility process from Standard Laplace (SLC).
        """
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def alpha(self):
        """
        Function to evaluate the ML estimator for alpha. 
        
        Alpha is the shape parameter of the pdf SLC. 

        Returns:
            float: Value for the ML estimator of alpha. 
        """
        x = np.linspace(0, 1, 1000)
        ml_a = self.beta, self.theta
        return ml_a

    def beta(self):
        """
        Function to evaluate the ML estimator for beta. 
        
        Beta represents impact from previous volatility step, similar to the generalisation from ARCH to GARCH. 

        Returns:
            float: Value for the ML estimator of beta. 
        """
        x = np.linspace(0, 1, 1000)
        ml_b = self.alpha, self.theta
        return ml_b
    
    def theta(self):
        """
        Function to evaluate the ML estimator for theta. Magnitude of the positive/netave news.  

        Returns:
            float: Value for the ML estimator of theta. 
        """
        x = np.linspace(0, 1, 1000)
        ml_t = self.alpha, self.beta
        return ml_t
