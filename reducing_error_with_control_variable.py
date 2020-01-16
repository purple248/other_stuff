import numpy as np
from numpy import random as rn
import scipy.stats as ss

# This example is about using control variable in simulation to reduce the variance in the estimation.
# here is estimation the price of asian call option that pays (average of s - k)+,
# with using control variable of regular (vanilla) call option, that can be calculated without simulation via black&schols.

r = 0.01     # interest rate
sigma = 0.4  # the stock std
T = 60/252   # exercise time, 252 = 1 year
S0 = 1       # stock price in time zero
K = 1.1      # strike price
M = 10**4    # number of simulation
N = 60       # number of points for simulation
h = T/N      # step size in time T

# for the price simulations
S = S0*np.ones((M, N+1))
dW = np.sqrt(h)*rn.randn(M, N)

# GBM analytic solution with simulation for the change in the stochastic part(dW):
for i in range(0, N):
    S[:, i+1] = S[:, i]*np.exp((r-(sigma**2)/2)*h + sigma*dW[:, i])
avg_per_simulation = np.mean(S, 1)-K
price_per_simulation = (avg_per_simulation > 0)*avg_per_simulation*np.exp(-r*T)

# monte carlo estimation for the price:
MC_price = [np.mean(price_per_simulation), np.std(price_per_simulation)/np.sqrt(M)]
print("price and error without controlled variable{}".format(MC_price))


# solution with control variable - vanilla call, will be:
# avg_Z = X - c(Y - E[Y])

# small simulation to estimate the correlation of the asian option and regular call with T
M = 5*10**3
Z = rn.randn(M,N)
S = np.ones((M,N+1))
S[:, 0] = S0*np.ones(M)
for i in range(0, N):
    S[:, i+1] = S[:, i]*np.exp((r-sigma**2/2)*h+sigma*np.sqrt(h)*Z[:,i])
avg_per_simulation_for_cor = np.mean(S, 1) - K
payoff_x_for_cor = (avg_per_simulation_for_cor > 0) * avg_per_simulation_for_cor * np.exp(-r * T)  # X
payoff_y_for_cor = np.exp(-r*T)*(S[:, -1] > K)*(S[:, -1] - K)  # Y

# the covariance matrix
q = np.cov(payoff_x_for_cor,payoff_y_for_cor)

# calculating c:
c = -q[0, 1] / q[1, 1]


# functions for Black&Schols price:
def d1(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))


def blsprice(type, S0, K, r, sigma, T):
    if type == "C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
        return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))


# the vanilla call price -> E[Y]:
V_BLS = blsprice('C', S0, K, r, sigma, T)


# the real simulation:
M = 45*10**3
Z = rn.randn(M,N)
S = np.ones((M,N+1))
S[:, 0] = S0*np.ones(M)
for i in range(0,N):
    S[:, i+1] = S[:, i]*np.exp((r-sigma**2/2)*h+sigma*np.sqrt(h)*Z[:, i])
avg_per_simulation = np.mean(S,1)-K
payoff_x = (avg_per_simulation > 0)*avg_per_simulation*np.exp(-r*T)  # X
payoff_y = np.exp(-r * T) * (S[:, -1] > K) * (S[:, -1] - K)  # Y

# calculating the estimated price for the asian option (X) using the controlled variable (Y)
# X*=X+c*(Y-E[Y]),  V_BLS=E[Y]
corrected = payoff_x+c*(payoff_y - V_BLS)

uncontrolled = [np.mean(payoff_x),np.std(payoff_x)/np.sqrt(M)]  # estimation without control variable
print("price and error, no controlled variable{}".format(uncontrolled))
controlled = [np.mean(corrected),np.std(corrected)/np.sqrt(M)]  # estimation with control variable
print("price and error with controlled variable (T){}".format(controlled))