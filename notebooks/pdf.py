# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3.9.12 ('scl-vol')
#     language: python
#     name: python3
# ---

from volatility_process import VolatilityProcess

from pdf_decomposed_standard_laplace import DecomposedSLC

# +
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# -

plt.style.use('seaborn-notebook')
mpl.rcParams['figure.facecolor'] = 'white'

# ### Volatility process 
# Under the constraint of 2 $\alpha$<1 - $\beta$ with zero shocks ($\theta$ = 0).

n = 2000
vol_exp = VolatilityProcess().exponential_process(n=n)
df_vol_exp = pd.DataFrame(vol_exp)
df_vol_exp.columns =["Volatility Process"]


# +
df_vol_exp.plot()
plt.title("Volatility process with zero shocks")
plt.xlabel("Time steps")
plt.show()


# -

# ## Varying $\alpha$
# Under the constraint of 2 $\alpha$<1 - $\beta$.
#
# The $\alpha$ parameter is the shape parameter and expected to have the most impact since it is in the denominator of the pdf. A low $\alpha$ gives high kurtosis while a low gives fatter tails.

# +
theta = 0.2
alphas=np.array([0.1, 0.15, 0.20, 0.25, 0.3])
rho = np.linspace(1,10,10000)

pdf_a = [DecomposedSLC(alpha=alpha).asymmetric_pdf(rho=rho, theta=theta) for alpha in alphas]
df_pdf_a = pd.DataFrame(np.transpose(pdf_a))
df_pdf_a.columns =['alpha='+str(alphas[0]), 'alpha='+str(alphas[1]), 'alpha='+str(alphas[2]), 'alpha='+str(alphas[3]), 'alpha='+str(alphas[4])]
# -

sns.lineplot(data=df_pdf_a).set(title='PDF varying alpha', xlabel = "Instances", ylabel = "PDF")
plt.ylim(0, 3)
plt.show()

# ## Varying $\beta$
# Under the constraint of $2\alpha<1-\beta$.
#
# The expectation is that the $\beta$ prameter works as a location parameter similar to the GARCH generalization of including a moving average component together with the autoregressive component. This means that a larger $\beta$ implies larger impact from previous steps and the opposite for a smaller $\beta$.

# +
theta = 0.2
betas = np.round(np.linspace(0.01,0.2, 10),2)
rho = np.linspace(1,10,10000)

pdf_b = [DecomposedSLC(beta=beta).asymmetric_pdf(rho=rho, theta=theta) for beta in betas]
df_pdf_b = pd.DataFrame(np.transpose(pdf_b))
df_pdf_b.columns =['beta='+str(betas[0]), 'beta='+str(np.round(betas[1],2)), 'beta='+str(betas[2]), 'beta='+str(betas[3]), 'beta='+str(betas[4]), 'beta='+str(betas[5]), 'beta='+str(betas[6]), 'beta='+str(betas[7]), 'beta='+str(betas[8]), 'beta='+str(betas[9]),]


# +

sns.lineplot(data=df_pdf_b).set(title='PDF varying beta', xlabel = "Instances", ylabel = "PDF")
plt.ylim(0, 2)
plt.show()
# -

#

# Varying $\theta$, $0 \leq \theta < 1$
#
# The parameter represents the positive and negative shocks in volatility. The impact will probably be limited since the impact is multiplied by $\alpha$ which often takes a small value due to the contraint of $2 \alpha<1-\beta$. 

# +
thetas = [0, 0.2, 0.4, 0.6, 0.8]
rho = np.linspace(1,10,10000)

pdf_t = [DecomposedSLC(theta=theta).asymmetric_pdf(rho=rho, theta=theta) for theta in thetas]
df_pdf_t = pd.DataFrame(np.transpose(pdf_t))
df_pdf_t.columns =['theta='+str(thetas[0]), 'theta='+str(thetas[1]), 'theta='+str(thetas[2]), 'theta='+str(thetas[3]), 'theta='+str(thetas[4])]

# -

sns.lineplot(data=df_pdf_t).set(title='PDF varying theta')
plt.ylim(0, 2)
plt.xlim(-20, 5000)
plt.show()
