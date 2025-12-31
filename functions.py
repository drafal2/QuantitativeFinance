
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


if __name__ == '__main__':
    x = np.array([15, 13, 17, 20, 27])
    ret = historical_volatility(x, False, True)
    print(ret)
