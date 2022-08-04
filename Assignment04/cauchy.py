"""
Plot Poisson rate posterior PDFs, and binomial alpha posterior PDFs, as
a demo of the UnivariateBayesianInference class.

Created Feb 27, 2015 by Tom Loredo
2018-03-08 Modified for Py-3, updated for BDA18
2020-02-26 Updated for BDA20; added __main__
"""

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from numpy import *
from scipy import stats, special, integrate

from univariate_bayes import UnivariateBayesianInference

    
class CauchyLocationInference(UnivariateBayesianInference):
    """
    Bayesian inference for a Poisson rate.
    """

    def __init__(self, obs, prior, r_l, r_u,d, nr=200):
        """
        Define a posterior PDF for a Cauchy location.
        Parameters
        ----------
        obs : observed data
        prior : const or function
            Prior PDF for the rate, as a constant for flat prior, or
            a function that can evaluate the PDF on an array
        r_u : float
            Upper limit on rate for evaluating the PDF
        d: float
            Known Cauchy scale
        """
        self.d = d
        self.r_l, self.r_u = r_l, r_u
        self.nr = nr
        self.rvals = linspace(r_l, r_u, nr)
        self.obs = obs

        # Pass the info to the base class initializer.
        super().__init__(self.rvals, prior, self.lfunc)

    def lfunc(self, rvals):
        """
        Evaluate the Poisson likelihood function on a grid of rates.
        """
        l=ones(self.nr)
        for x in self.obs:
          x0 = (x - rvals)/self.d
          l=l/(1+x0**2)
        return l
    
if __name__ == '__main__':

    ion()


