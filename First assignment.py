
# coding: utf-8

# In[1]:


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[2]:


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"


# In[3]:


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']


# In[4]:


dataset = pandas.read_csv(url, names=names)


# In[5]:


print(dataset.shape)


# In[6]:


print(dataset.head(10))


# In[7]:


print(dataset.describe())


# In[8]:


print(dataset.groupby('class').size())
#shows no. of instances that belong to each class in column class


# In[9]:


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[10]:


dataset.hist()


# In[12]:


plt.show()


# In[13]:


# scatter plot matrix to see relationship between inputs
scatter_matrix(dataset)
plt.show()


# In[14]:


# Split-out validation dataset ,first we will check validation and then fit it to the algorithm then we will check the algorithm with training set
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[15]:


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# In[16]:


models = LogisticRegression()
names = 'LR'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(models, X_train, Y_train, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % (names, cv_results.mean(), cv_results.std())
print(msg)


# In[22]:


fig=plt.figure()
plt.boxplot(cv_results)
plt.show()


# In[24]:


fig=plt.figure()
plt.pie(cv_results)
plt.show()


# In[25]:


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[26]:


lr=LogisticRegression()
lr.fit(X_train, Y_train)
prediction = lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

