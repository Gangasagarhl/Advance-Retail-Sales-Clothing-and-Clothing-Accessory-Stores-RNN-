#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,SimpleRNN,GRU
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import r2_score,max_error,mean_squared_error,median_absolute_error
from sklearn.preprocessing import MinMaxScaler


# In[2]:


# Reading data from csv
data = pd.read_csv(r"C:\Users\RAGHAVENDRA\Desktop\FINAL_TF2_FILES\DATA\RSCCASN.csv",index_col='DATE',parse_dates=True)


# In[3]:


data.info()


# In[4]:


data.head()


# In[5]:


# Visualising data
data.plot(figsize=(12,8))
plt.show()


# ### Data Preprocessing

# #### Splitting data

# In[6]:


# I want 10% of the whole data has to be splitted to train and test
length = 0.1
split_length =len(data) -  int(len(data)*length)
train_data = data.iloc[:split_length]
test_data = data.iloc[split_length:]

len(train_data),len(test_data)


# #### Scaling data using MinMaxScaler

# In[7]:


#scaling data
scaler = MinMaxScaler()
scale_train = scaler.fit_transform(train_data)
scale_test  = scaler.transform(test_data) 


# #### Preparing Time series generator data for training as well as validation data

# In[8]:


def timeserieGenerator(length=12,batch_size=1):
    train_generator = TimeseriesGenerator(scale_train,scale_train,length = length,batch_size = batch_size)
    
    validation_generator = TimeseriesGenerator(scale_test,scale_test,length = length,batch_size = batch_size)
    
    return train_generator,validation_generator,length

length = int(input("Enter the length:"))
batch_size = int(input("Enter Batch Size:"))

train_generator,validation_generator,length = timeserieGenerator(length,batch_size)


# ### Building Model

# In[9]:


def building_and_fitting_model(model_type,length=12,n_features = 1):
    
    
    model1 =  Sequential()
    model1.add(model_type(50,activation = 'relu',input_shape=(length,n_features)))
    model1.add(Dense(1))
    model1.compile(optimizer='adam',loss = 'mse')

    print(model1.summary())
    ES = EarlyStopping(monitor = 'val_loss',mode = 'min',patience=5)
    model1.fit_generator(train_generator,validation_data = validation_generator,epochs = 300,callbacks = [ES])
    
    print(str(model_type),":\n")
    df = pd.DataFrame(model1.history.history)
    df.plot()
    
    return model1


# In[10]:



def forecast(to_be_forecasted,model):
    forecast = []
    first_eval_batch = scale_train[-length:]
    current_eval_batch = first_eval_batch.reshape((1,length,batch_size))
    
    for i in range(to_be_forecasted):
        
        prediction = model.predict(current_eval_batch)[0]
        forecast.append(prediction)
        current_eval_batch = np.append(current_eval_batch[:,1:,:],[[prediction]],axis=1)
        
    forecast = scaler.inverse_transform(forecast)
    
    return forecast
        
    
    
    


# #### LSTM

# In[11]:


model_LSTM = building_and_fitting_model(LSTM,length = length , n_features = batch_size)
forecast_points = forecast(len(scale_test),model_LSTM)
test_data["LSTM"] = forecast_points


# ### SimpleRNN
# 

# In[12]:


len(scale_test)


# In[13]:


model_SRNN = building_and_fitting_model(SimpleRNN,length = length , n_features = batch_size)
forecast_points = forecast(len(scale_test),model_SRNN)
test_data["SimpleRNN"] = forecast_points


# ### GRU

# In[14]:


model_GRU = building_and_fitting_model(GRU,length = length , n_features = batch_size)
forecast_points = forecast(len(scale_test),model_GRU)
test_data["GRU"] = forecast_points


# In[16]:


test_data.plot(figsize=(12,8))


# ### Evaluating model using reccursion metrics

# In[22]:


def max_error_value(true,predicted):
    return max_error(true,predicted)

def r2score(true,predicted):
    return r2_score(true,predicted)

def mean_squared_error_value(true,predicted):
    return mean_squared_error(true,predicted)

def mean_squared_error_value(true,predicted):
    return mean_squared_error(true,predicted)



def evaluating_models():
    #Printing Max Error
    print("Max Error from LSTM:",max_error_value(test_data[['RSCCASN']],test_data[['LSTM']]))
    print("Max Error from SimpleRNN:",max_error_value(test_data[['RSCCASN']],test_data[['SimpleRNN']]))
    print("Max Error from GRU:",max_error_value(test_data[['RSCCASN']],test_data[['GRU']]))
    print("\n\n")
    #Mean Squared Error
    print("Mean Squared Error from LSTM: ",mean_squared_error_value(test_data[['RSCCASN']],test_data[['LSTM']]))
    print("Mean Squared Error from SimpleRNN: ",mean_squared_error_value(test_data[['RSCCASN']],test_data[['SimpleRNN']]))
    print("Mean Squared Error from GRU: ",mean_squared_error_value(test_data[['RSCCASN']],test_data[['GRU']]))
    print("\n\n")
    #r2_score
    rscr = 0
    model = 'LSTM'
    
    #LSTM
    rscr = r2score(test_data[['RSCCASN']],test_data[['LSTM']])
    print("r2_score From LSTM:",rscr)
    
    #SimpleRNN
    temp = r2score(test_data[['RSCCASN']],test_data[['SimpleRNN']])
    print("r2_score From SimpleRNN:",temp)
    
    if temp>rscr:
        rscr = temp
        model = 'SimpleRNN'
        
    #GRU    
    temp = r2score(test_data[['RSCCASN']],test_data[['GRU']])
    print("r2_score From GRU:",temp)
    
    if temp>rscr:
        rscr = temp
        model = 'GRU'
        
    print('\n\nBest Model Among All Is: ',model ,"With r2_score: ",rscr)
    
evaluating_models()
        
    


# ##### Based on the above, we are using SimpleRNN for predicting or forecasting for an year's data

# ### Forecasting results with the trained model of SimpleRNN

# ##### Note: More and more you forecast,introducing of noise is too much into data.

# In[20]:


scaled_data_for_forecasting = scaler.fit_transform(data)


# In[24]:


train_data.tail()


# In[32]:


period = int(input('Enter the number of years to be forecasted:'))
period *= 12
forecasting_result = forecast(period,model_SRNN)


# In[33]:


forecating_index = pd.date_range(start='2017-02-01',periods=period,freq='MS')
forecating_index


# In[37]:


forecast_dataframe = pd.DataFrame(data = forecasting_result,index = forecating_index,
                                 columns = ['Forecast'])


# In[41]:


# Forecasted dataframe
forecast_dataframe


# ##### plotting in different plots

# In[42]:


train_data.plot()
forecast_dataframe.plot()


# ##### Plotting in same axis

# In[47]:


ax = train_data.plot(figsize=(12,8))
forecast_dataframe.plot(ax=ax)
plt.xlim('2015-01-01','2021-01-01')

