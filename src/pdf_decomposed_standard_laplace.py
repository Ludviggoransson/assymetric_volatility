"""
Assymmetric Power Laplace GARCH. Probability density function of from a Standard Laplace (SLC). 
"""

import numpy as np

class DecomposedSLC():

    def __init__(self, rho:np.array, alpha: float, beta: float, theta: float):
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def assymmetric_pdf(self):
        rho_t = self.rho[1:]
        rho_lag = self.rho[0:-1]

        numerator_pos = rho_t-(1+self.beta*rho_lag)
        denumerator_pos = self.alpha*(1+self.theta)*rho_lag

        numerator_neg = rho_t-(1+self.beta*rho_lag)
        denumerator_neg = self.alpha*(1-self.theta)*rho_lag

        numerator = np.exp(-(numerator_pos/denumerator_pos))-np.exp(-(numerator_neg/numerator_pos))
        denumerator = 2*self.alpha*self.theta*rho_lag
        pdf = numerator/denumerator
        return pdf


    def symmetric_pdf(self):
        pdf = self.rho
        return pdf
