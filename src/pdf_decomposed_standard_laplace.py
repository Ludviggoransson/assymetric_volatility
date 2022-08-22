"""
Probability density function from a Standard Laplace (SLC) represented by a difference of two exponential processes. 
"""

import numpy as np
from volatility_process import VolatilityProcess

class DecomposedSLC(VolatilityProcess):
    """
    Class to present symmetric and asymmetric probability density functions. 

    Args:
        VolatilityProcess (Class): Class to generate symmetric and asymmetric volatility processes. 
                                    Mainly to inherit default values of the parameters in the pdf. 
    """

    def asymmetric_pdf(self, rho:np.array, theta:float=0.5)->np.array:
        """
        Asymmetric probability density function, i.e. theta≠0. 

        Args:
            rho (np.array): Sequence of values that represents the volatility process. 
            theta (float, optional): Asymmetric parameter. Defaults to 0.5.

        Returns:
            np.array: probability density function in the case of theta≠0.
        """

        rho_t = rho[1:]
        rho_lag = rho[0:-1]

        numerator_sub = rho_t-(1+self.beta*rho_lag)

        denumerator_pos = self.alpha*(1+theta)*rho_lag
        denumerator_neg = self.alpha*(1-theta)*rho_lag

        numerator = np.exp(-(numerator_sub/denumerator_pos))-np.exp(-(numerator_sub/denumerator_neg))
        denumerator = 2*self.alpha*theta*rho_lag
        pdf = numerator/denumerator
        return pdf


    def symmetric_pdf(self, rho:np.array)->np.array:
        """
        Symmetric probability density funciton, i.e. theat=0.

        Args:
            rho (np.array): Sequence of values that represents the volatility process. 

        Returns:
            np.array: probability density function in the case of theta≠0.
        """
        rho_t = rho[1:]
        rho_lag = rho[0:-1]

        multiplicator = rho_t-self.beta*rho_lag-1

        numerator_sub = rho_t-(1+self.beta*rho_lag)
        denumerator_sub = self.alpha*rho_lag

        numerator = multiplicator*np.exp(-(numerator_sub/denumerator_sub))
        denumerator = (self.alpha*rho_lag)**2
        pdf_symmetric = numerator/denumerator

        return pdf_symmetric
        
if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    theta = 0.5
    beta = 0.01
    alphas=np.array([0.1, 0.2, 0.3, 0.4])
    n = 100
    rho = np.linspace(1.4255,10,80000)
    pdf = [DecomposedSLC(alpha=alpha).asymmetric_pdf(rho=rho) for alpha in alphas]
    df_pdf = pd.DataFrame(np.transpose(pdf))
    df_pdf.columns =['alpha='+str(alphas[0]), 'alpha='+str(alphas[1]), 'alpha='+str(alphas[2]), 'alpha='+str(alphas[3])]
    sns.lineplot(data=df_pdf).set(title='PDF varying alpha')
    plt.ylim(0)
    n = 1
