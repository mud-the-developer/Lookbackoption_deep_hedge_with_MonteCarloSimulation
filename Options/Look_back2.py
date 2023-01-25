import numpy as np
from scipy.stats import norm

def lbcall(S, S_min, T, r, sig, q=0.02):
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

    small_t = list(range(0,1000))
    small_t.reverse()
    small_t =np.asarray(small_t)+T/1000
    ttt=np.ones([101,1000]) 
    small_t = small_t*ttt
    small_t=small_t*T/1000
    small_t=np.transpose(small_t)
    a1 = (np.log(S/S_min)+(r-q+0.5*sig**2)*small_t)/(sig*np.sqrt(small_t))
    a2 = a1-(sig*np.sqrt(small_t))
    a3 = (np.log(S/S_min)+(q-r+0.5*sig**2)*small_t)/(sig*np.sqrt(small_t))
    gamma1 = (2*(r-q-0.5*sig**2)*np.log(S/S_min))/(sig**2)
    
    cost=S*np.exp(-q*small_t)*(norm.cdf(a1)-(sig**2)*norm.cdf(-a1)/(2*(r-q)))
    cost=cost-S_min*np.exp(-r*small_t)*(norm.cdf(a2)
                                  -np.exp(gamma1)*norm.cdf(-a3)*sig**2/(2*(r-q)))

    return cost,a1
    

def lbput(S, S_max, T, r, sig, q=0.02):
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
    small_t = list(range(0,1000))
    small_t.reverse()
    small_t =np.asarray(small_t)+T/1000
    ttt=np.ones([101,1000])
    small_t = small_t*ttt
    small_t=small_t*T/1000
    small_t=np.transpose(small_t)
    T=small_t
    b1 = (np.log(S_max/S)+(q-r+0.5*sig**2)*T)/(sig*np.sqrt(T))
    b2 = b1-(sig*np.sqrt(T))
    b3 = (np.log(S_max/S)+(r-q-0.5*sig**2)*T)/(sig*np.sqrt(T))
    gamma2 = (2*(r-q-0.5*sig**2)*np.log(S_max/S))/(sig**2)
    
    cost=S_max*np.exp(-r*T)*(norm.cdf(b1)
                             -np.exp(gamma2)*norm.cdf(-b3)*sig**2/(2*(r-q)))
    cost=cost+S*np.exp(-q*T)*((sig**2)*norm.cdf(-b2)/(2*(r-q)) - norm.cdf(b2))
    return cost
    
