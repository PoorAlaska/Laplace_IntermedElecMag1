##########################################################################
# PHYS 211 Intermediate Electricity and Magetism -1
# Numerical Evaluation of the Two-Dimensional Laplace Equations
#
# PROGRAM:  Primary task is to evaluate the linear average of neighboring 
#           points in a mesh with a certain size
# INPUT:    None
# CREATED:  1/29/15
# AUTHOR:   Tahoe Schrader
#                         
##############################################################################

from __future__ import division
import matplotlib
#matplotlib.use('MacOSX') # Not sure what backend is for linux
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

#############################################################################
## Variables are given parameters to begin with 
#############################################################################
numxbins = 32
numybins = 32
stepsize = 1/128
maxdiff = 1e-5
diff = 1.0 
imax = 10000

# I decided to go with the number of bins given by the professor in the 
# skeleton. So, as can be seen, the dimensionality of the system will only be
# two dimensions. There is some number of possible i,j combinations in which the
# i and j can only equal 32 as a maximum. This number takes longer for the 
# computer to compute. Lowering 32 would make things go faster. The maximum 
# iterations for this job is just fine and tolerance was found fairly quickly.  

##############################################################################
## The most important part of the code is develop a system with which to 
## numerically compute iterations. This is the toughest part, but there is a
## skeleton we were given with which to follow. 
##############################################################################
## Initialize the mesh with default values of zero
V = np.zeros((numxbins,numybins),float)

## Set boundary conditions for the problem
for i in range(0,numybins,1): 
    V[i,numxbins-1] = 1.0
print V 

#for j in range(0,numybins,1): 
#    V[numxbins-1,j] = 1.0
#print V 

# I commented out the above boundary condition just because I like the way the
# graphs looked with only the one boundary. It shows, however, that I'm capable
# of adding more if I wanted to. Essentially, this boundary is saying that one 
# edge is permanently set to be equal to 1.0. I could change this number to any
# number if i wanted though. 

## Iterate until the solution converges 
while diff > maxdiff: 
    diff = 0.0
    ## Set an iteration counter 
    iterations=0.
    # Iterate over the two dimensional space 
    for i in range(1,numxbins-1,1): 
        for j in range(1,numxbins-1,1):
            # Calculate the new value of the potential at this point
            # and the difference with respect to the previous value
            Vsave = (V[i-1,j]+V[i+1,j]
                      +V[i,j-1]+V[i,j+1])/4.0
            diffsave = abs(Vsave-V[i,j])
            if diffsave > diff:
                diff = diffsave
                iterations += 1.
            # Set the potential to the newly calculated value and increment the counter
            V[i,j] = Vsave 
print V

#############################################################################
## Plotting the function is then straightforward...
#############################################################################

x = np.linspace(0, numxbins * stepsize, numxbins)
y = np.linspace(0, numybins * stepsize, numybins)
X,Y = np.meshgrid(x,y)

plt.figure()
surf = plt.contourf(X,Y,V,32, rstride=1, cstride=1, cmap=cm.cool)
plt.colorbar(surf)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Laplace Solver')
plt.show()

plt.figure()
plt.gca(projection='3d')
cs = plt.contourf(X,Y,V,32, rstride=1, cstride=1, cmap=cm.cool)
plt.colorbar(cs)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Laplace Solver in 3D!!!')
plt.show()# Laplace_IntermedElecMag1
