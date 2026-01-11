
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import math


# TODO: prepare docs in readable format

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

def generate_gbm_paths(s0, T, r, sigma, time_steps_num, path_num, normal_dist_vars=None):

    dt = T / time_steps_num
    if normal_dist_vars is None:
        normal_dist_vars = np.random.normal(0, 1, (path_num, time_steps_num))
    
    paths = np.zeros((path_num, time_steps_num + 1))
    paths[:, 0] = s0

    for i in range(1, time_steps_num + 1):
        paths[:, i] = paths[:, i - 1] + paths[:, i - 1] * r * dt + sigma * paths[:, i - 1] * np.sqrt(dt) * normal_dist_vars[:, i - 1]
        
    return paths.T

def generate_money_market_numeraire_paths(T, r, time_steps_num, path_num):

    dt = T / time_steps_num
    paths = np.zeros((path_num, time_steps_num + 1))
    paths[:, 0] = 1

    for i in range(1, int(time_steps_num) + 1):
        paths[:, i] = paths[:, i - 1] * (1 + r * dt)  # dg = r*g*dt   ->   g_t+1 - g_t = r*g_t*dt   ->   g_t+1 = g_t + r*g_t*dt   ->   g_t+1 = (1+r*dt) * g_t
    
    return paths.T


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

def bsm_option_price(s0, K, T, r, sigma, q=0.0, type='call'):
    # q - dividend yield, when stock does not pay dividend then q=0.0

    d1 = (np.log(s0/K) + (r - q + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if type == 'call':
        price = s0 * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = np.exp(-r * T) * K * norm.cdf(-d2) - s0 * np.exp(-q*T) * norm.cdf(-d1)
        
    return price, d1, d2

def binomial_coefficient(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

def main():
    pass


if __name__ == '__main__':
    main()
