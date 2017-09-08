import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gym
from gym import envs
print(envs.registry.all())

def savitzky_golay(y, window_size=5, order=1, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')





def set_maze_series():
    with open("Maze_simulation_true_online_no_heuristics.pkl", "rb") as f:
        series = pickle.load(f)
    with open("Maze_simulation_true_online_heuristics_hard2.pkl", "rb") as f:
        series2 = pickle.load(f)       
    with open("Maze_simulation_true_online_heuristics_hard_model_based.pkl", "rb") as f:
        series3 = pickle.load(f)
    with open("Maze_simulation_true_online_static_heuristics.pkl", "rb") as f:
        series4 = pickle.load(f)
    return series,series2,series3,series4


def set_m3d_series():
    with open("M3D_simulation_true_online_no_heuristics.pkl", "rb") as f:
        series = pickle.load(f)
    with open("M3D_simulation_true_online_heuristics_hard.pkl", "rb") as f:
        series2 = pickle.load(f)
    with open("M3D_simulation_true_online_heuristics_hard_model_based.pkl", "rb") as f:
        series3 = pickle.load(f)
    return series,series2,series3



def set_mc_series():
    with open("Mountain_car_simulation_true_online_no_heuristics.pkl", "rb") as f:
        series = pickle.load(f)
    with open("Mountain_car_simulation_true_online_heuristics_hard.pkl", "rb") as f:
        series2 = pickle.load(f)
    with open("Mountain_car_simulation_true_online_heuristics_hard_model_based.pkl", "rb") as f:
        series3 = pickle.load(f)
    return series,series2,series3


def set_mc_series_rep():
    with open("Mountain_car_simulation_replacing_no_heuristics.pkl", "rb") as f:
        series = pickle.load(f)
    with open("Mountain_car_simulation_replacing_heuristics_hard.pkl", "rb") as f:
        series2 = pickle.load(f)
    with open("Mountain_car_simulation_replacing_heuristics_soft.pkl", "rb") as f:
        series3 = pickle.load(f)
    return series,series2,series3


series,series2,series3,series4=set_maze_series()

print len(series)
print len(series2)
print len(series3)   
print len(series4)
   
mean_s = np.average(series,axis=0)
std = np.std(series,axis=0)/np.sqrt(len(series))


mean_hard = np.average(series2,axis=0)
std_hard = np.std(series2,axis=0)/np.sqrt(len(series2))

mean_soft = np.average(series3,axis=0)
std_soft = np.std(series3,axis=0)/np.sqrt(len(series3))

mean_heur = np.average(series4,axis=0)
std_heur = np.std(series4,axis=0)/np.sqrt(len(series4))


filtered_mean_s =savitzky_golay(mean_s)
filtered_mean_hard =savitzky_golay(mean_hard)
filtered_mean_soft =savitzky_golay(mean_soft)
filtered_mean_heur =savitzky_golay(mean_heur)

#serie= pd.Series(mean_s)
#serie.plot()
plt.semilogy()
plt.plot(range(len(mean_s)),filtered_mean_s,color="b")

minim = np.argmin(mean_s)
print np.min(mean_s)
print std[minim]

plt.plot(range(len(mean_hard)),filtered_mean_hard,color="r")

minim = np.argmin(mean_hard)
print np.min(mean_hard)
print std_hard[minim]

plt.plot(range(len(mean_soft)),filtered_mean_soft,color="g")

print np.min(mean_soft)
print std_soft[minim]


plt.plot(range(len(mean_heur)),filtered_mean_heur,color="black")

minim = np.argmin(mean_heur)
print np.min(mean_heur)
print std[minim]



plt.fill_between(range(len(filtered_mean_s)),filtered_mean_s-2*std,filtered_mean_s+2*std, color="b",alpha=0.2)

plt.fill_between(range(len(filtered_mean_hard)),filtered_mean_hard-2*std_hard,filtered_mean_hard+2*std_hard, color="r",alpha=0.2)

plt.fill_between(range(len(filtered_mean_soft)),filtered_mean_soft-2*std_soft,filtered_mean_soft+2*std_soft, color="g",alpha=0.2)

plt.fill_between(range(len(filtered_mean_heur)),filtered_mean_heur-2*std_heur,filtered_mean_heur+2*std_heur, color="black",alpha=0.2)


plt.show()