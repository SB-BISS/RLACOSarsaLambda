import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#For mountain car, cumulative Reward can be calculated simply by 499 - l


with open("Mountain_car_H_TrueOnline_hard.pkl", "rb") as f:
   result_h = pickle.load(f)
   

with open("Mountain_car_H_TrueOnline_soft.pkl", "rb") as f:
   result_h2 = pickle.load(f)


print len(result_h)
print len(result_h2)


def find_parameters(result_vec):
    min_val = 500000
    par = []
    index = 0
    for i in range(len(result_vec)):
        if result_vec[i]['results'][-1]< min_val:
            min_val=result_vec[i]['results'][-1] # the 20th episode
            index = i
            print min_val
            print index
            par = result_vec[i]['parameters']
            
    return par,min_val,index 

    

pars_h,min_rep_h,index_h = find_parameters(result_h)

pars_h2,min_rep_h2,index_h2 = find_parameters(result_h2)

#print pars_h
#print pars_h2


plt.plot(result_h[index_h]['results'], 'b')
plt.plot(result_h2[index_h2]['results'],'r')
plt.show()

print result_h[index_h]
print result_h2[index_h2]





with open("Mountain_car_simulation_true_online_no_heuristics.pkl", "rb") as f:
   series = pickle.load(f)


with open("Mountain_car_simulation_true_online_heuristics_hard.pkl", "rb") as f:
   series2 = pickle.load(f)


with open("Mountain_car_simulation_true_online_heuristics_soft.pkl", "rb") as f:
   series3 = pickle.load(f)

   
   
mean_s = np.average(series,axis=0)
std = np.std(series,axis=0)/np.sqrt(len(series))


mean_hard = np.average(series2,axis=0)
std_hard = np.std(series2,axis=0)/np.sqrt(len(series2))

mean_soft = np.average(series3,axis=0)
std_soft = np.std(series3,axis=0)/np.sqrt(len(series3))


#serie= pd.Series(mean_s)
#serie.plot()
plt.plot(range(200),mean_s,color="b")
plt.plot(range(200),mean_hard,color="r")
plt.plot(range(200),mean_soft,color="g")

plt.fill_between(range(200),mean_s-2*std,mean_s+2*std, color="b",alpha=0.2)

plt.fill_between(range(200),mean_hard-2*std,mean_hard+2*std_hard, color="r",alpha=0.2)

plt.fill_between(range(200),mean_soft-2*std,mean_soft+2*std_soft, color="g",alpha=0.2)
plt.show()