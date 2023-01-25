import numpy as np
from scipy.stats import norm

def bscall(S, K, T, r, sig, q=0.02):
    """_summary_
    Black-Scholes Call Option
    Args:
        S : Stock price
        K : Strike price
        T : Time to maturity 100/365
        r : interest rate
        q : dividend yield 0.02
        sigma : volatility

    Returns:
        premium: x=[premium]+[cost]+[SS]
    """
    d1 = (np.log(S/K)+(r-q+0.5*sig**2)*T)/(sig*np.sqrt(T))
    d2 = (np.log(S/K)+(r-q-0.5*sig**2)*T)/(sig*np.sqrt(T))
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2), d1
    
def bsput(S, K, T, r, sig, q=0.02):
    """_summary_
    Black-Scholes Put Option
    Args:
        S : Stock price
        K : Strike price
        T : Time to maturity 100/365
        r : interest rate
        sigma : volatility

    Returns:
        premium: x=[premium]+[cost]+[SS]
    """
    d1 = (np.log(S/K)+(r-q+0.5*sig**2)*T)/(sig*np.sqrt(T))
    d2 = (np.log(S/K)+(r-q-0.5*sig**2)*T)/(sig*np.sqrt(T))
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)