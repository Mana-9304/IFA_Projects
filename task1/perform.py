#!/usr/bin/env python
# coding: utf-8

# In[1]:


import webbrowser
from fyers_apiv3 import fyersModel
import datetime


# In[2]:


client_id = 'Y93EC2KB25-100'
secret_key = '32VZTT1H8V'
redirect_uri = 'https://trade.fyers.in/'
current_date = datetime.datetime.now().date()


# In[3]:


#Input parameters
grant_type = "authorization_code"


# In[4]:


response_type = "code"
state = "sample"


# In[5]:


### Connect to the sessionModel object here with the required input parameters
appSession = fyersModel.SessionModel(client_id = client_id, redirect_uri = redirect_uri,response_type=response_type,state=state,secret_key=secret_key,grant_type=grant_type)


# In[6]:


### Make  a request to generate_authcode object this will return a login url which you need to open in your browser from where you can get the generated auth_code
generateTokenUrl = appSession.generate_authcode()


# In[7]:


print((generateTokenUrl))
webbrowser.open(generateTokenUrl,new=1)


# In[ ]:


### After succesfull login the user can copy the generated auth_code over here and make the request to generate the accessToken
auth_code = input("Enter Auth Code: ")
appSession.set_token(auth_code)
response = appSession.generate_token()


# In[ ]:


### There can be two cases over here you can successfully get the acccessToken over the request or you might get some error over here. so to avoid that have this in try except block
try:
    access_token = response["access_token"]
    print("token: ",access_token)
except Exception as e:
    print(e,response)  ## This will help you in debugging then and there itself like what was the error and also you would be able to see the value you got in response variable. instead of getting key_error for unsuccessfull response.


# In[ ]:


fyers = fyersModel.FyersModel(token=access_token,is_async=False,client_id=client_id)


# In[ ]:


#Get details about your account
response = fyers.get_profile()
print(response)


# In[ ]:


#save to txt file
with open("client_id_mine.txt",'w') as file:
    file.write(client_id)


# In[ ]:


with open("access_token_mine.txt",'w') as file:
    file.write(access_token)

