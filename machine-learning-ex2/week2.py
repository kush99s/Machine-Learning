
# coding: utf-8

# In[78]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize


# In[79]:


def plotdata(X,y):
    pos= X[np.where(y==1)]
    neg= X[np.where(y==0)]
    fig,ax=plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")
    return (fig,ax)


# In[80]:


def costFunction(theta,X,y):
    m = len(y) 
    J =((np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-(1-y)*(np.log(1-sigmoid(np.dot(X,theta))))))/m)
    grad = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    return (J, grad)


# In[81]:


def sigmoid(z):
    return (1.0/(1 + np.e**(-z)))


# In[82]:


def predict(theta,X):
    return np.where(np.dot(X,theta)> .5,1,0)
    """
    Given a vector of parameter results and training set X,
    returns the model prediction for admission. If predicted
    probability of admission is greater than .5, predict will
    return a value of 1.
    """


# In[83]:


def costfunctionreg(X,y,theta,reg_param):
    m=len(y)
    J=((np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-(1-y)*(np.log(1-sigmoid(np.dot(X,theta))))))/m + (reg_param/m)*np.sum(theta**2))
    #regularizing cost function 
    grad_0 = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    #regularized grad
    grad_reg= grad_0+ (reg_param/m)*theta
    # Replace gradient for theta_0 with non-regularized gradient
    grad_reg[0]=grad_0[0]
    return (J,grad_reg)


# In[84]:


def mapFeatureVector(X1,X2):
    """
    Feature mapping function to polynomial features. Maps the two features
    X1,X2 to quadratic features used in the regularization exercise. X1, X2
    must be the same size.returns new feature array with interactions and quadratic terms
    """
    
    degree = 6
    output_feature_vec = np.ones(len(X1))[:,None]

    for i in range(1,7):
        for j in range(i+1):
            new_feature = np.array(X1**(i-j)*X2**j)[:,None]
            output_feature_vec = np.hstack((output_feature_vec,new_feature))
   
    return (output_feature_vec)


# In[85]:


def plotDecisionBoundary(theta,X,y):
    """X is asssumed to be either:
        1) Mx3 matrix where the first column is all ones for the intercept
        2) MxN with N>3, where the first column is all ones
    """
    fig, ax = plotdata(X[:,1:],y)
    """
    if len(X[0]<=3):
        # Choose two endpoints and plot the line between them
        plot_x = np.array([min(X[:,1])-2,max(X[:,2])+2])
        ax.plot(plot_x,plot_y)
        ax.legend(['Admitted','Fail','Pass'])
        ax.set_xbound(30,100)
        ax.set_ybound(30,100)
    else:
    """

    # Create grid space
    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v)))
    
    # Evaluate z = theta*x over values in the gridspace
    for i in range(len(u)):
        for j in range(len(v)):
            z[i][j] = np.dot(mapFeatureVector(np.array([u[i]]),np.array([v[j]])),theta)
    
    # Plot contour
    ax.contour(u,v,z,levels=[0])

    return (fig,ax)


# In[86]:


data = pd.read_csv('ex2data1.txt', names=['x1','x2','y'])
X = np.asarray(data[["x1","x2"]])
y = np.asarray(data["y"])
fig, ax = plotdata(X, y)
ax.legend(['Admitted', 'Not admitted'])
fig.show()


# In[87]:


# Add intercept term to x and X_test
X = np.hstack((np.ones_like(y)[:,None],X))
initial_theta = np.zeros(3)
cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): \n', cost)
print('Gradient at initial theta (zeros): \n',grad)


# In[88]:


arr=[-24,0.2,0.2]
test_theta= np.array(arr)


# In[89]:


cost, grad = costFunction(test_theta, X, y)
print(cost)
print(grad)


# In[90]:


res = minimize(costFunction,initial_theta, method='Newton-CG',args=(X,y), jac=True,  options={'maxiter':400,'disp':True})
theta = res.x
print('Cost at theta found by minimize: \n', res.fun)
print('theta: \n', theta)


# In[91]:


prob = sigmoid(np.dot([1,95,85],theta))
print('For a student with scores 45 and 85, we predict an ', 'admission probability of ', prob)
# Compute accuracy on our training set
p = predict(theta, X)
print('Train Accuracy: \n', np.mean(p==y)*100)


# In[92]:


plotDecisionBoundary(theta, X, y)

