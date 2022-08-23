"""
Evaluation of Maximum Likelihood estimators of alpha and beta. 
"""
from re import A
import numpy as np
from volatility_process import VolatilityProcess

class MLEstimator(VolatilityProcess):
    """
    Class to generate Maximum Likehood estimators from decomposing a Standard Classical Laplace to a difference of two exponential processes.

    I use numerical optimisation to find the ML estimators. 
    """

    def ml_estimator(self, n:int):
        """
        Function to evaluate the ML estimator for alpha and beta. 
        
        Alpha is the shape parameter of the pdf SLC. 

        Returns:
            float: Value for the ML estimator of alpha. 
        """

        rho = VolatilityProcess().exponential_process(n=n)
        rho_t = rho[1:]
        rho_lag = rho[0:-1]
        C = 0.0000001
        alpha = np.full(n, np.nan)
        beta = np.full(n, np.nan)

        beta[0] = self.beta/2
        alpha[0] = self.alpha/2
        alpha_denumerator = 2*n*rho_lag
        for i in range(1, n):
            alpha_numerator = rho_t-beta[i-1]*rho_lag-1
            beta_denumerator = rho_t-beta[i-1]*rho_lag-1

            alpha[i] = alpha[i-1] - C*(alpha[i-1]-sum(alpha_numerator/alpha_denumerator))
            beta[i] = beta[i-1] - C*(sum(rho_lag/beta_denumerator)-(n/alpha[i-1]))
        
        return alpha, beta

if __name__ == "__main__":
    import pandas as pd
    x = MLEstimator().ml_alpha(n=10000)
    
    pd.DataFrame(x[0]).plot()
    pd.DataFrame(x[1]).plot()
    n = 1
