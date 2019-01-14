
# coding: utf-8

# ## Import Libraries

# In[70]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[71]:


iris=datasets.load_iris()

df=pd.DataFrame(iris['data'],columns=iris['feature_names'])
df['Flower']=pd.DataFrame(iris.target)

data=df.drop(['petal length (cm)','petal width (cm)'],axis=1)
data


# ## Shuffle the Data

# In[72]:


data=data.sample(frac=1).reset_index(drop=True)
#data['sepal length (cm)'].iloc[np.random.randint(100)]
data.head()


# Split data in train and test

# In[73]:


train, test = train_test_split(data, test_size=0.2)
#cl_df=pd.DataFrame({"col":[1]})

#c=["yellow",purple]
plt.scatter(train['sepal length (cm)'],train['sepal width (cm)'],c=train.Flower)
plt.scatter(test['sepal length (cm)'],test['sepal width (cm)'],c="red",s=25)
plt.show()


# ### Define Number of Neighbours

# In[74]:


k=5
train.drop('Flower',axis=1).iloc[1]
dist=((train.drop('Flower',axis=1).iloc[1]-train.drop('Flower',axis=1))**2).sum(axis=1)
# idx=(dist.sort_values()[0:k]).index
# train.Flower[idx].value_counts()
# #test.head()
# test.iloc[1:5]
#dist
test


# ### Apply KNN Algortihm

# In[75]:


plt.scatter(train['sepal length (cm)'],train['sepal width (cm)'],c=train.Flower)
plt.scatter(test['sepal length (cm)'],test['sepal width (cm)'],c="red")
selected_classes=[]
for i in range(len(test)):
    #print(i)
    #dist=(test["sepal length (cm)"].iloc[i]-train["sepal length (cm)"])**2
    dist=(np.sqrt((test.drop('Flower',axis=1).iloc[i]-train.drop('Flower',axis=1))**2)).sum(axis=1)
    #plt.scatter(test.iloc[i].name,test["sepal length (cm)"].iloc[i],s=100,marker="D")
    plt.pause(0.05)
    
    #print(dist)
    dist=dist.sort_values()[0:k]
    indexes=dist.index
    
    pnt=plt.scatter(test['sepal length (cm)'].iloc[i],test['sepal width (cm)'].iloc[i],s=150,marker="o",facecolors='none', edgecolors='r')
    nbr=plt.scatter(train["sepal length (cm)"][indexes],train['sepal width (cm)'][indexes],s=150,marker="D",facecolors='none', edgecolors='g')
    
    plt.pause(1)
    #time.sleep(0.01)
    pnt.remove()
    nbr.remove()

    selected_classes.append(train.Flower[indexes].value_counts().idxmax())
#print(selected_classes)
test["Predicted Classes"]=np.asarray(selected_classes)
#pd.DataFrame(pd.DataFrame(selected_classes),ignore_index=True)
test





(test.Flower==test["Predicted Classes"]).value_counts()




