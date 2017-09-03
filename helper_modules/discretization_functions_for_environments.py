

def discretize_mountain_car():    
        # state discretization for the mountain car problem
        xdiv  = (0.6-(-1.2))   / 10.0
        xpdiv = (0.07-(-0.07)) / 5.0
        
        x = np.arange(-1.5,0.5+xdiv,xdiv)
        xp= np.arange(-0.07,0.07+xpdiv,xpdiv)

        N=np.size(x)
        M=np.size(xp)

        states=[] #zeros((N*M,2)).astype(Float32)
        index=0
        for i in range(N):    
            for j in range(M):
                states.append([x[i], xp[j]])
                
		return np.array(states)