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
    with open("Maze_simulation_true_online_no_heuristics_final.pkl", "rb") as f:
        dict0 = pickle.load(f)
        series = dict0["series"]
        times = dict0["times"]
    with open("Maze_simulation_true_online_heuristics_hard_final.pkl", "rb") as f:
        dict0 = pickle.load(f)
        series2 = dict0["series"]
        times2 = dict0["times"]
    with open("Maze_simulation_true_online_heuristics_hard_model_based_final.pkl", "rb") as f:
        dict0 = pickle.load(f)
        series3 = dict0["series"]
        times3 = dict0["times"]
    with open("Maze_simulation_true_online_static_heuristics_final.pkl", "rb") as f:
        dict0 = pickle.load(f)
        series4 = dict0["series"]
        times4 = dict0["times"]
    with open("Maze_simulation_true_online_static_pheromone_heuristic.pkl", "rb") as f:
        dict0 = pickle.load(f)
        series5 = dict0["series"]
        times5 = dict0["times"]
    
            
    return series,series2,series3,series4,series5,times,times2,times3,times4,times5



def set_m3d_series():
    with open("M3D_simulation_true_online_no_heuristics_final.pkl", "rb") as f:
        dict0 = pickle.load(f)
        series = dict0["series"]
        times = dict0["times"]
    with open("M3D_simulation_true_online_heuristics_hard_final.pkl", "rb") as f:
        dict2 = pickle.load(f)
        series2 = dict2["series"]
        times2= dict2["times"]
    with open("M3D_simulation_true_online_heuristic_hard_model_based_final.pkl", "rb") as f:
        dict3 = pickle.load(f)
        series3 = dict3["series"]
        times3= dict3["times"]
        
    with open("M3D_simulation_true_online_static_heuristic2.pkl", "rb") as f:
        dict4 = pickle.load(f)
        series4 = dict4["series"]
        times4= dict4["times"]
           
    with open("M3D_simulation_true_online_static_pheromone_heuristic.pkl", "rb") as f:
        dict5 = pickle.load(f)
        series5 = dict5["series"]
        times5= dict5["times"]
        
    return series,series2,series3,series4,series5, times,times2,times3,times4,times5


def set_mc_series_potential():
    with open("Mountain_car_simulation_true_online_no_heuristics_final.pkl", "rb") as f:
        dict0 = pickle.load(f)
        series = dict0["series"]
        times = dict0["times"]
    with open("Mountain_car_simulation_true_online_heuristics_hard_final.pkl", "rb") as f:
        dict2 = pickle.load(f)
        series2 = dict2["series"]
        times2= dict2["times"]
    with open("Mountain_car_simulation_true_online_potential_pheromone_heuristic2.pkl", "rb") as f:
        dict3 = pickle.load(f)
        series3 = dict3["series"] 
        times3= dict3["times"]
        
    with open("Mountain_car_simulation_true_online_potential_pheromone_heuristic3.pkl", "rb") as f:
        dict4 = pickle.load(f)
        series4 = dict4["series"]
        times4= dict4["times"]
        
    return series,series2,series3,series3,series4,times,times2,times3,times4,times4



def set_mc_series():
    with open("Mountain_car_simulation_true_online_no_heuristics_final.pkl", "rb") as f:
        dict0 = pickle.load(f)
        series = dict0["series"]
        times = dict0["times"]
    with open("Mountain_car_simulation_true_online_heuristics_hard_final.pkl", "rb") as f:
        dict2 = pickle.load(f)
        series2 = dict2["series"]
        times2= dict2["times"]
    with open("Mountain_car_simulation_true_online_heuristics_hard_model_based_final.pkl", "rb") as f:
        dict3 = pickle.load(f)
        series3 = dict3["series"]
        times3= dict3["times"]
    with open("Mountain_car_simulation_true_online_static_heuristic_final.pkl", "rb") as f:
        dict4 = pickle.load(f)
        series4 = dict4["series"]
        times4= dict4["times"]
    
    with open("Mountain_car_simulation_true_online_static_pheromone_heuristic.pkl", "rb") as f:
        dict5 = pickle.load(f)
        series5 = dict5["series"]
        times5= dict5["times"]
        
    return series,series2,series3,series4,series5,times,times2,times3,times4,times5


def set_mc_series_rep():
    with open("Mountain_car_simulation_replacing_no_heuristics.pkl", "rb") as f:
        series = pickle.load(f)
    with open("Mountain_car_simulation_replacing_heuristics_hard.pkl", "rb") as f:
        series2 = pickle.load(f)
    with open("Mountain_car_simulation_replacing_heuristics_soft.pkl", "rb") as f:
        series3 = pickle.load(f)
    return series,series2,series3


series,series2,series3,series4,series5, t,t2,t3,t4,t5=set_mc_series_potential()


def find_best_results(time,series):
    
    best_times = []
    total_times =[]
    best_indexes = []
    best_values = []
    inds = 0
    for seq in time:
        
        seq_series = series[inds]
        
        index = np.argmin(seq_series)
        min_value = seq_series[index]
                
        best_indexes.append(index) 
        best_times.append(np.sum(seq[0:index]))
        total_times.append(np.sum(seq))
        best_values.append(min_value)
        inds+=1
        
    dict_pack = {"values":best_values, "indexes":best_indexes, "times":best_times, "execution_times":total_times} 
    return dict_pack




def print_results(ts,seriess):
    dict_pack = find_best_results(ts,seriess)
    
    values= dict_pack["values"]
    indexes= dict_pack["indexes"]
    times= dict_pack["times"]
    total_time = dict_pack["execution_times"]
    
    mean_values = np.average(values)
    std_values = np.std(values,axis=0)/np.sqrt(len(values))

    mean_indexs = np.average(indexes)

    mean_indexes = np.average(indexes)
    std_indexes = np.std(indexes,axis=0)/np.sqrt(len(indexes))

    mean_times = np.average(times)
    std_times = np.std(times,axis=0)/np.sqrt(len(times))
    
    mean_execution_times= np.average(total_time)
    std_exec_times = np.std(total_time,axis=0)/np.sqrt(len(total_time))
    
    print "LEN VALUES" + str(len(values))
    print "MEAN values " + str(mean_values) + "Interval " + str(std_values)
    print "Mean indexes " + str(mean_indexes) + "Interval " + str(std_indexes)
    print "Mean times " + str(mean_times) + "interval " + str(std_times)
    print "Average Execution Times" + str(mean_execution_times) + "interval" + str(std_exec_times)


print "BEST RESULTS Standard"
print_results(t,series)
print "END BEST RESULTS Standard"

print "BEST RESULTS Backward"
print_results(t2,series2)
print "END BEST RESULTS Backward"


print "BEST RESULTS Forward"
print_results(t3,series3)
print "END BEST RESULTS Forward"

print "BEST RESULTS Heuristic"
print_results(t4,series4)
print "END BEST RESULTS Heuristic"

print "BEST RESULTS Heuristic Pher"
print_results(t5,series5)
print "END BEST RESULTS Heuristic Pher"


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


mean_heur_p = np.average(series5,axis=0)
std_heur_p = np.std(series5,axis=0)/np.sqrt(len(series5))

filtered_mean_s =savitzky_golay(mean_s)
filtered_mean_hard =savitzky_golay(mean_hard)
filtered_mean_soft =savitzky_golay(mean_soft)
filtered_mean_heur =savitzky_golay(mean_heur)

filtered_mean_heur_p =savitzky_golay(mean_heur_p)

lw = 3
#serie= pd.Series(mean_s)
#serie.plot()
plt.semilogy()
plt.plot(range(len(mean_s)),filtered_mean_s,color="b", linewidth=lw)

print "standard Algorithm"
minim = np.argmin(mean_s)
print np.min(mean_s)
print std[minim]
print minim


plt.plot(range(len(mean_hard)),filtered_mean_hard,color="r", linewidth=lw)

print "backward"
minim = np.argmin(mean_hard)
print np.min(mean_hard)
print std_hard[minim]
print minim #average minimum

plt.plot(range(len(mean_soft)),filtered_mean_soft,color="g", linewidth=lw)

print "forward"
minim = np.argmin(mean_soft)
print np.min(mean_soft)
print std_soft[minim]
print minim

#plt.plot(range(len(mean_heur)),filtered_mean_heur,color="black", linewidth=lw)

print "heuristic"
minim = np.argmin(mean_heur)
print np.min(mean_heur)
print std[minim]
print minim


plt.plot(range(len(mean_heur_p)),filtered_mean_heur_p,color="orange", linewidth=lw)

print "heuristic_p"
minim = np.argmin(mean_heur_p)
print np.min(mean_heur_p)
print std[minim]
print minim


a=plt.fill_between(range(len(filtered_mean_s)),filtered_mean_s-2*std,filtered_mean_s+2*std, color="b",alpha=0.4)

b=plt.fill_between(range(len(filtered_mean_hard)),filtered_mean_hard-2*std_hard,filtered_mean_hard+2*std_hard, color="r",alpha=0.4)

c=plt.fill_between(range(len(filtered_mean_soft)),filtered_mean_soft-2*std_soft,filtered_mean_soft+2*std_soft, color="g",alpha=0.4)

#d=plt.fill_between(range(len(filtered_mean_heur)),filtered_mean_heur-2*std_heur,filtered_mean_heur+2*std_heur, color="black",alpha=0.4, label="HA-SARSA($\lambda$)")

e=plt.fill_between(range(len(filtered_mean_heur_p)),filtered_mean_heur_p-2*std_heur_p,filtered_mean_heur_p+2*std_heur_p, color="orange",alpha=0.4, label="HA-SARSA($\lambda$)")


plt.legend(["Approximated Sarsa($\lambda$)","AP-HARL-BACKWARD","RS-BACKWARD","RS-SARSA($\lambda$)+Pheromone"])

#plt.legend( ["Approximated Sarsa($\lambda$)","AP-HARL-BACKWARD","AP-HARL-FORWARD","HA-SARSA($\lambda$)","HA-SARSA($\lambda$)+Pheromone"])
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='black', linestyle='--')
plt.minorticks_on()
plt.show()