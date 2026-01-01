
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def historical_volatility(ts, if_returns=False, if_alternative_approach=False):
    """
    Calculates historical volatility based on given prices or log-returns. Two alternative approaches 
    are implemented, giving the same result. Does not take into account time scaling so returned 
    volatility is computed as a volatility on time interval of given prices/ returns
    """
    ts = np.asarray(ts)
    if not if_returns:
        ts = np.log(ts[1:] / ts[:-1])
    
    n = len(ts)
    
    if if_alternative_approach:
        vol = np.sqrt(1/(n-1) * np.sum(ts**2) - 1/(n*(n-1)) * np.sum(ts)**2)
    else:
        avg = np.average(ts)
        vol = np.sqrt(1/(n-1) * np.sum((ts - avg) ** 2))
                
    return vol

def eq_option_pricing_binomial_tree(s0, K, T, r, sigma, N):
    """Function utilized in binomial tree option pricing

    Args:
        s0 (float): current underlying price
        K (float): strike price
        T (float): maturity of option in years
        r (float): risk-free rate
        sigma (float): volatility of an underlying
        N (int): number of time steps from 0 to T

    Returns:
        float: option price
        np.ndarray: array with binomial tree of underlying prices
    """
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    s_tree = s0 * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1, 1))
    V = np.maximum(s_tree - K, 0)

    for i in range(int(N) - 1, -1, -1):
        V = discount * (p * V[:-1] + (1 - p) * V[1:])
        
    return V[0], s_tree

def main():
    return None


if __name__ == '__main__':
    main()
