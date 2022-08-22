"""
Assymmetric Power Laplace GARCH. Probability density function of from a Standard Laplace (SLC). 
"""

import numpy as np
from volatility_process import VolatilityProcess

class DecomposedSLC(VolatilityProcess):

    def assymmetric_pdf(self, rho:np.array, theta:float=0.5):

        rho_t = rho[1:]
        rho_lag = rho[0:-1]

        numerator_sub = rho_t-(1+self.beta*rho_lag)

        denumerator_pos = self.alpha*(1+theta)*rho_lag
        denumerator_neg = self.alpha*(1-theta)*rho_lag

        numerator = np.exp(-(numerator_sub/denumerator_pos))-np.exp(-(numerator_sub/denumerator_neg))
        denumerator = 2*self.alpha*theta*rho_lag
        pdf = numerator/denumerator
        return pdf


    def symmetric_pdf(self, rho:np.array):
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
    pdf = [DecomposedSLC(alpha=alpha).symmetric_pdf(rho=rho) for alpha in alphas]
    df_pdf = pd.DataFrame(np.transpose(pdf))
    df_pdf.columns =['alpha='+str(alphas[0]), 'alpha='+str(alphas[1]), 'alpha='+str(alphas[2]), 'alpha='+str(alphas[3])]
    sns.lineplot(data=df_pdf).set(title='PDF varying alpha')
    plt.ylim(0)
    n = 1
