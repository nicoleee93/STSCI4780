"""
Functions for computing and plotting components of a
Bayesian normal-normal conjugate model.
"""

from numpy import *
import matplotlib as mpl
from matplotlib.pyplot import *
from scipy import stats, integrate
import numpy.testing as npt


class FigLRAxes:
    """
    A figure with two ordinate axes (left and right) sharing a common
    abscissa axis.
    
    In matplotlib lingo, this is a two-scale plot using twinx().
    """

    def __init__(self, figsize=(8,6), l=0.15, r=0.85):
        self.fig = figure(figsize=figsize)
        # Left and right axes:
        self.leftax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=l, right=r)
        self.rightax = self.leftax.twinx()
        # Use thicker frame lines.
        self.leftax.patch.set_lw(1.25)  # thicker frame lines
        self.rightax.patch.set_lw(1.25)  # thicker frame lines
        # Leave with the left axes as current.
        self.fig.sca(self.leftax)

    def left(self):
        self.fig.sca(self.leftax)
        return self.leftax

    def right(self):
        self.fig.sca(self.rightax)
        return self.rightax


def plot_norm_norm1(mus, mu0, w0, dbar, w):
    """
    Plot the prior, likelihood, and posterior for a
    normal-normal conjugate Bayesian model.
    """
    # Prior as a frozen RV object:
    prior = stats.norm(mu0, w0)
    # Strictly speaking, the likelihood is not a PDF;
    # we'll take it as proportional to a normal PDF,
    # with unit proportionality constant.
    like = stats.norm(dbar, w)

    # *** YOUR COMPUTATION OF POSTERIOR HERE ***
    B=w*w/(w*w+w0*w0)
    posterior=stats.norm(dbar+B*(mu0-dbar),w*sqrt(1-B))
    fig = FigLRAxes()

    # Prior, posterior against left axis:
    # *** YOUR CODE HERE ***
    plot(mus, prior.pdf(mus), 'b-', lw=2, label='Prior')
    plot(mus, posterior.pdf(mus), 'k-', lw=2, alpha=.7, label='Post')
    xlabel(r'$\alpha$')
    ylabel('PDF')
    legend(loc='upper left')
    # Likelihood against the right axis:
    # *** YOUR CODE HERE***
    fig.right()
    plot(mus,like.pdf(mus), 'r:', lw=4, alpha=.5, label='Like')
    ylabel('Like')
    legend(loc='upper right')
    # Use the same ylim, so the height of the likelihood curve
    # reflects its width relative to the prior & posterior:
    fig.left()
    l, u = ylim()
    fig.right()
    ylim(l, u)


def plot_norm_norm2(mus, mu0, w0, dbar, w):
    """
    Plot the prior, likelihood, and posterior for a
    normal-normal conjugate Bayesian model.
    """
    # Prior as a frozen RV object:
    prior = stats.norm(mu0, w0)
    # Strictly speaking, the likelihood is not a PDF;
    # we'll take it as proportional to a normal PDF,
    # with unit proportionality constant.
    like = stats.norm(dbar, w)

    # *** YOUR NUMERICAL COMPUTATION OF POSTERIOR HERE ***
    pvals=prior.pdf(mus)
    lvals=like.pdf(mus)
    mlike = integrate.trapz(pvals*lvals, mus)
    ppdf = pvals*lvals/mlike
    fig = FigLRAxes()

    # Prior, posterior against left axis:
    # *** YOUR CODE HERE ***
    plot(mus, pvals, 'b-', lw=2, label='Prior')
    plot(mus, ppdf, 'k-', lw=2, alpha=.7, label='Post')
    xlabel(r'$\alpha$')
    ylabel('PDF')
    legend(loc='upper left')
    # Likelihood against the right axis:
    fig.right()
    # *** YOUR CODE HERE***
    plot(mus,lvals, 'r:', lw=4, alpha=.5, label='Like')
    ylabel('Like')
    legend(loc='upper right')
    
    # Use the same ylim, so the height of the likelihood curve
    # reflects its width relative to the prior & posterior:
    fig.left()
    l, u = ylim()
    fig.right()
    ylim(l, u)
    
    return ppdf


def test_norm():
    """
    Test normalization of the posterior (using global quantities).
    """
    # *** YOUR CODE HERE ***
    mus = linspace(0., 20., 500)
    ppdf = plot_norm_norm2(mus, 10., 2., 3., 1.)
    npt.assert_allclose(integrate.trapz(ppdf, mus), 1.)


# The following is for students who choose to debug the module
# by running it in a shell with IPython.

if __name__ == '__main__':
    # This puts matplotlib in "interactive" mode, which is useful
    # for command-line interaction, e.g., for testing this module
    # by running it with IPython.
    ion()

    # This should generate two plots.
    mus = linspace(0., 20., 500)
    ppdf = plot_norm_norm2(mus, 10., 2., 3., 1.)
    plot_norm_norm1(mus, 10., 2., 3., 1.)
    npt.assert_allclose(integrate.trapz(ppdf, mus), 1.)


