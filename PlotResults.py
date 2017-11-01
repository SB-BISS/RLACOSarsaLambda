import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#For mountain car, cumulative Reward can be calculated simply by 499 - l

''' Mountain Car
with open("Mountain_car_H_TrueOnline_hardmodelFalse.pkl", "rb") as f:
   result_h = pickle.load(f)
   
with open("Mountain_car_H_TrueOnline_softmodelFalse.pkl", "rb") as f:
   result_h2 = pickle.load(f)

with open("Mountain_car_H_TrueOnline_hardmodelTrue.pkl", "rb") as f:
   result_hm = pickle.load(f)
   
with open("Mountain_car_H_TrueOnline_softmodelTrue.pkl", "rb") as f:
   result_h2m = pickle.load(f)
'''


with open("M3D_H_TrueOnline_hardmodelFalse.pkl","rb") as f:
    result_h = pickle.load(f)


with open("M3D_H_TrueOnline_hardmodelTrue.pkl","rb") as f:
    result_h2 = pickle.load(f)

with open("M3D_SH_TrueOnline_hardmodelTrue.pkl","rb") as f:
    result_hm = pickle.load(f)



#with open("Dyna_H_TrueOnline_hardmodelFalse.pkl","rb") as f:
#    result_h = pickle.load(f)


#with open("M3D_H_TrueOnline_hardmodelTrue.pkl","rb") as f:
#    result_h = pickle.load(f)

with open("Dyna_SH_TrueOnline_hardmodelFalse.pkl","rb") as f:
    result_hm = pickle.load(f)



print len(result_h)
print len(result_h2)
print len(result_hm)
#print len(result_h2m)


def find_parameters(result_vec):
    min_val = 500000
    par = []
    index = 0
    for i in range(len(result_vec)):
        if result_vec[i]['cumulative_sum']< min_val:
            min_val=result_vec[i]['cumulative_sum'] # the 20th episode
            index = i
            print min_val
            print index
            par = result_vec[i]['parameters']
            
    return par,min_val,index 

    

pars_h,min_rep_h,index_h = find_parameters(result_h)

pars_h2,min_rep_h2,index_h2 = find_parameters(result_h2)

pars_hm,min_rep_hm,index_hm = find_parameters(result_hm)

#pars_h2m,min_rep_h2m,index_h2m = find_parameters(result_h2m)

#print pars_h
#print pars_h2


plt.plot(result_h[index_h]['results'], 'b')
plt.plot(result_h2[index_h2]['results'],'r')
plt.plot(result_hm[index_hm]['results'], 'g')
#plt.plot(result_h2m[index_h2m]['results'],'c')
plt.show()

print "backward"
print result_h[index_h]

print "forward"
print result_h2[index_h2]

print "hard"
print result_hm[index_hm]
#print "soft"
#print result_h2m[index_h2m]



'''
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
plt.show()'''