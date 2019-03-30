
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


# In[3]:


def warmupexercise():
    return np.eye(5)


# In[4]:


def plotdata(X,Y):
    fig,ax= plt.subplots() #creating empty figure
    ax.plot(X,Y,'rx',markersize=10)
    ax.set_xlabel("Population of city in 10,000s")
    ax.set_ylabel("Profit in $10,000s")  #labeling axes
    return fig


# In[25]:


def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history= np.zeros(num_iters) #to save value of J after every iteration
    for i in range (num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)
        J_history[i]=computecost(X,y,theta)  #calculating cost after each descent
        print("Cost function: ",J_history[i])
    return (theta,J_history)    


# In[7]:


def featurenormalize(X):
    return np.divide((X-np.mean(X,axis=0)),np.std(X,axis=0))


# In[8]:


def computecost(X,y,theta):
    m=len(y)
    J=(np.sum((np.dot(X,theta)-y)**2))/(2*m) #computing cost
    return J


# In[10]:


print ("Running warmup exercise")
print(warmupexercise())


# In[16]:


print("plotting data")
data = pd.read_csv("ex1data1.txt",names=["X","y"]) #divides data in 2 parts of column named as X and y
x= np.array(data.X)[:,None]
y=np.array(data.y)
fig=plotdata(x,y)
fig.show()


# In[26]:


theta=np.zeros(2)
iterations=1500
alpha = 0.01
ones=np.ones_like(x)
X= np.hstack((ones,x))
J=computecost(X,y,theta)
print("cost with theta as 0:",J)
theta,hist= gradientDescent(X,y,theta,alpha,iterations)
print("gradeint is ",theta[0],"\n",theta[1])


# In[31]:


plt.plot(x,y,'rx',x,np.dot(X,theta),'b-')
plt.legend(['Training Data','Linear Regression'])
plt.show()         


# In[32]:


predict1 = np.dot([1, 3.5],theta) # takes inner product to get y_bar
print('For population = 35,000, we predict a profit of ', predict1*10000)

predict2 = np.dot([1, 7],theta)
print('For population = 70,000, we predict a profit of ', predict2*10000)
input('Program paused. Press enter to continue.\n');
print('Visualizing J(theta_0, theta_1) ...\n')


# In[39]:


# Grid over which we will calculate J 
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i],theta1_vals[j]])
        J_vals[i][j] = computecost(X,y,t)
fig = plt.figure()
ax = plt.subplot(111,projection='3d')
Axes3D.plot_surface(ax,theta0_vals,theta1_vals,J_vals,cmap=cm.coolwarm)
plt.show()
fig = plt.figure()
ax = plt.subplot(111)
plt.contour(theta0_vals,theta1_vals,J_vals)         

