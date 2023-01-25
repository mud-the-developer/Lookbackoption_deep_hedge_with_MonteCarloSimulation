import numpy as np
from scipy.stats import norm

def lbcall_premium(S, S_min, T, r, sig, q=0.02):
    """_summary_

    Args:
        S (_type_): Stock price
        S_min (_type_): Stock_min
        T (_type_): dt
        r (_type_): interest rate
        sig (_type_): volatility
        q (float, optional): dividend yield 0.02. Defaults to 0.02.

    Returns:
        cost
    """
    a1 = (np.log(S/S_min)+(r-q+0.5*sig**2)*T)/(sig*np.sqrt(T))
    a2 = a1-(sig*np.sqrt(T))
    a3 = (np.log(S/S_min)+(q-r+0.5*sig**2)*T)/(sig*np.sqrt(T))
    gamma1 = (2*(r-q-0.5*sig**2)*np.log(S/S_min))/(sig**2)
    
    cost=S*np.exp(-q*T)*(norm.cdf(a1)-(sig**2)*norm.cdf(-a1)/(2*(r-q)))
    cost=cost-S_min*np.exp(-r*T)*(norm.cdf(a2)
                                  -np.exp(gamma1)*norm.cdf(-a3)*sig**2/(2*(r-q)))
    return cost
    

def lbput_premium(S, S_max, T, r, sig, q=0.02):
    """_summary_

    Args:
        S (_type_): Stock price
        S_max (_type_): Stock_max
        T (_type_): Time to maturity 30/365
        r (_type_): interest rate
        sig (_type_): volatility
        q (float, optional): dividend yield 0.02. Defaults to 0.02.

    Returns:
        cost
    """
    b1 = (np.log(S_max/S)+(q-r+0.5*sig**2)*T)/(sig*np.sqrt(T))
    b2 = b1-(sig*np.sqrt(T))
    b3 = (np.log(S_max/S)+(r-q-0.5*sig**2)*T)/(sig*np.sqrt(T))
    gamma2 = (2*(r-q-0.5*sig**2)*np.log(S_max/S))/(sig**2)
    
    cost=S_max*np.exp(-r*T)*(norm.cdf(b1)
                             -np.exp(gamma2)*norm.cdf(-b3)*sig**2/(2*(r-q)))
    cost=cost+S*np.exp(-q*T)*((sig**2)*norm.cdf(-b2)/(2*(r-q)) - norm.cdf(b2))
    return cost
    
