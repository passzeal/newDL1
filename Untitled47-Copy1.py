#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


# In[2]:


boston = tf.keras.datasets.boston_housing


# In[3]:


dir(boston)


# In[4]:


boston=boston.load_data()


# In[5]:


print(type(boston))


# In[6]:


(train_data,train_targets),(train_data,train_targets)=boston


# In[7]:


data = pd.DataFrame(train_data)


# In[8]:


data.head()


# In[9]:


data.tail()


# In[10]:


data.columns


# In[11]:


data.describe()


# In[12]:


data.dtypes


# In[13]:


data.info()


# In[14]:


data['PRICE']=train_targets


# In[15]:


print(data.head())


# In[16]:


print(data.shape)


# In[17]:


data.isnull().sum()


# In[18]:


import seaborn as sns
sns.distplot(data.PRICE)


# In[19]:


sns.boxplot(data.PRICE)


# In[20]:


correlation=data.corr()
correlation.loc['PRICE']


# In[21]:


import matplotlib.pyplot as plt
fig,axes=plt.subplots(figsize=(15,12))
sns.heatmap(correlation,square=True,annot=True)


# In[22]:


feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT','New']


# In[23]:


data.columns = feature_names 


# In[24]:


data['PRICE']=train_targets


# In[25]:


plt.figure(figsize = (20,5))
features = ['LSTAT','RM','PTRATIO']
for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = data[col]
    y = data.PRICE
    plt.scatter(x, y, marker='o')
    plt.title("Variation in House prices")
    plt.xlabel(col)
    plt.ylabel('"House prices in $1000"')


# In[26]:


X = data.iloc[:,:-1]
y= data.PRICE


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(train_data, train_targets, test_size=0.2, random_state=42)


# In[29]:


mean = X_train.mean(axis=0)
std = X_train.std(axis=0)


# In[30]:


X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# In[31]:


from sklearn.linear_model import LinearRegression


# In[32]:


regressor = LinearRegression()
#Fitting the model
regressor.fit(X_train,y_train)
# Model Evaluation
#Prediction on the test dataset
y_pred = regressor.predict(X_test)
# Predicting RMSE the Test set results


# In[33]:


from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print(rmse)


# In[34]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)


# In[35]:


# Neural Networks
#Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[36]:


import keras
from keras.layers import Dense, Activation,Dropout
from keras.models import Sequential


# In[37]:


get_ipython().system('pip install ann_visualizer')
get_ipython().system('pip install graphviz')


# In[38]:


pip install --upgrade ann_visualizer


# In[39]:


# Build your model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam',loss ='mean_squared_error',metrics=['mae'])


# In[40]:


from keras.utils import plot_model


# In[41]:


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[42]:


history = model.fit(X_train, y_train, epochs=100, validation_split=0.05)


# In[43]:


pip install plotly


# In[44]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scattergl(y=history.history['loss'],
name='Train'))
fig.add_trace(go.Scattergl(y=history.history['val_loss'],
name='Valid'))
fig.update_layout(height=500, width=700,
xaxis_title='Epoch',
yaxis_title='Loss')
fig.show()


# In[45]:


y_pred = model.predict(X_test)
mse_nn, mae_nn = model.evaluate(X_test, y_test)
print('Mean squared error on test data: ', mse_nn)
print('Mean absolute error on test data: ', mae_nn)


# In[46]:


from sklearn.metrics import mean_absolute_error
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[47]:


y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)


# In[48]:


print('Mean squared error on test data: ', mse_lr)
print('Mean absolute error on test data: ', mae_lr)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)


# In[ ]:




