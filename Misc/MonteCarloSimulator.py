import numpy as np


def MonteCarloSimulator(S0,K,T,r,sigma,N,M):
    """_summary_

    Args:
        S0 : Current stock price
        K : Strike price
        T : Time to maturity 100/365
        r : interest rate
        sigma : volatility
        N : dt = T/N
        M : Number of stocks
        [M,N]
    """
    
    dt = T/N
    rdt = r*dt
    sigsdt = sigma*np.sqrt(dt)
    S=np.empty([M,N+1])
    rv=np.random.normal(rdt,sigsdt,[M,N])
    
    for i in range(M):
        S[i,0]= S0
        for j in range(N):
            S[i,j+1]=S[i,j]*(1+rv[i,j])
            
    return S
