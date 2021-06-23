#Social Ski-Driver (SSD) optimization algorithm
# More details about the algorithm are in [please cite the original paper (below)]
#Alaa Tharwat, Thomas Gabel, "Parameters optimization of support vector machines for imbalanced data using social ski driver algorithm"
#Neural Computing and Applications, pp. 1-14, 2019
#Tharwat, A. & Gabel, T. Neural Comput & Applic (2019). https://doi.org/10.1007/s00521-019-04159-z

import numpy as np
import matplotlib.pyplot as plt

Max_iterations=50  # Maximum Number of Iterations
correction_factor = 2.0 # Correction factor
inertia = 1.0 # Ineritia Coeffecient
swarm_size = 20 # Number of particles
LB=-10*np.ones((1,2))
UB=10*np.ones((1,2))

Xrange=UB[0,0]-LB[0,0]
Yrange=UB[0,1]-LB[0,1]
Dim=UB.shape[1]

NoRuns=100  # Number of runs

# Initial Positions
ConvergenceCurve=np.zeros((Max_iterations,NoRuns))
for r in range(NoRuns):
    Swarm=np.zeros((swarm_size,2))
    PreviouBest=float('inf')* np.ones((swarm_size,3))
    # Initial best value so far
    GlobalBestSolution=float('inf')
    BestSolns=float('inf')* np.ones((swarm_size,2))
    # Initial velocity
    Velocity=np.zeros((swarm_size,2))
    Fval=np.zeros((swarm_size))
    for i in range(swarm_size):
        Swarm[i,0]= np.random.random()*Xrange+LB[0,0]
        Swarm[i,1]= np.random.random()*Yrange+LB[0,1]

    for iter in range(Max_iterations):
        # Calculating fitness value for all particles    
        for i in range(swarm_size):
            Swarm[i,0]=Swarm[i,0]+Velocity[i,0]
            Swarm[i,1]=Swarm[i,1]+Velocity[i,1]
            
            # Update Position Bounds
            Swarm[i,0] = max(Swarm[i,0],LB[0,0])
            Swarm[i,0] = min(Swarm[i,0],UB[0,0])
            Swarm[i,1] = max(Swarm[i,1],LB[0,1])
            Swarm[i,1] = min(Swarm[i,1],UB[0,1])
            
            # The fitness function (Rastrigin) F(x,y)=20+(x^2-10cos(2\pix))+(y^2-10cos(2\pi y))
            X = Swarm[i,0]
            Y = Swarm[i,1]
            Fval[i]=20+X**2-10.*np.cos(2*3.14159*X)+Y**2-10*np.cos(2*3.14159*Y)
            
            if Fval[i]<PreviouBest[i,2]:
                PreviouBest[i,0]=Swarm[i,0] # Update the position of the first dimension
                PreviouBest[i,1]=Swarm[i,1] # Update the position of the second dimension
                PreviouBest[i,2]=Fval[i]          # Update best value
                
        a=2.0-iter*(float((2)/float(Max_iterations))) # a decreases linearly fron 2 to 0
        
        # Search for the global best solution
        (Gbest,idxGbest) = min((v,i) for i,v in enumerate(PreviouBest[:,2]))
        #    or
        #    Gbest= PreviouBest[:,2].min()
        #    idxGbest=np.argmin(PreviouBest[:,2])
        
        SortedElements=np.sort(PreviouBest[:,2])
        idxSortedElements=np.argsort(PreviouBest[:,2])
        # Find the mean of the best three solutions
        M=np.zeros((Dim))
        for i in range(Dim):
            M[i]=np.mean(Swarm[idxSortedElements[0:3],i])
    
        r1=np.random.random()  #r1 is a random number in [0,1]
        r2=np.random.random()  # r2 is a random number in [0,1]
        
        A1=2*a*r1-a     
        C1=2*r2         
        
        # Updating velocity vectors        
        for i in range(swarm_size):
            if np.random.random()<0.5:
                for j in range(Dim):
                    Velocity[i,j]=(a)*np.sin(np.random.random())*(PreviouBest[i,j] - Swarm[i,j])+(2.0-a)*np.sin(np.random.random())*(M[j]- Swarm[i,j])   # velocity component
            else:
                for j in range(Dim):
                    Velocity[i,j]=(a)*np.cos(np.random.random())*(PreviouBest[i,j] - Swarm[i,j])+(2.0-a)*np.cos(np.random.random())*(M[j]- Swarm[i,j])   # velocity component
        
        ConvergenceCurve[iter,r]=SortedElements[0]

# Plot the convergence curves of all runs
idx=range(Max_iterations)
fig= plt.figure()

#3-plot
ax=fig.add_subplot(111)
for i in range(NoRuns):
    ax.plot(idx,ConvergenceCurve[:,i])
plt.title('Convergence Curve of the social ski-driver algorithm', fontsize=12)
plt.ylabel('Fitness value')
plt.xlabel('Iterations')
plt.show()