#!/usr/bin/env python
# coding: utf-8

# In[16]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import time
from scipy.stats import linregress
import citipy
from citipy import citipy
from config import api_key
import json
import os

output_data_file = "output/cities.csv"


# In[2]:


#Set ranges
lat_range = (-90, 90)
lng_range = (-180, 180)


# In[ ]:


#Create lists for latitudes/longitudes and cities
lat_lngs = []
cities = []


# In[3]:


# Use .random to select random latitudes/longitudes
lats = np.random.uniform(low=-90.000, high=90.000, size=1500)
lngs = np.random.uniform(low=-180.000, high=180.000, size=1500)

#Use zip to combine latitudes/longitudes and put in list
lat_lngs = zip(lats, lngs)


# In[ ]:


# Using the list made, identify nearest city for latitudes/longitudes combinations
for lat_lng in lat_lngs:
    near_city = citipy.nearest_city(lat_lng[0], lat_lng[1]).city_name
    
    # If the city is unique, then add it to a our cities list
    if near_city not in cities:
        cities.append(near_city)


# In[4]:


#Set up lists to append data into
city_list = []
max_temp_list = []
humidity_list = []
cloud_list = []
wind_speed_list = []
country_code_list = []
date_list = []
lat_list = []
lng_list = []


# In[24]:


index = 0
set_count = 1 
print("Begin")

base_url = "http://api.openweathermap.org/data/2.5/weather?"
imperial = "imperial"

#Use f string to set up query url
query_url = f"{base_url}appid={api_key}&units={imperial}&q="

# For each city name in cities list, do below things...
for city in (cities):
    try:
        response_json = requests.get(query_url + city).json()
        city_list.append(response_json["name"])
        cloud_list.append(response_json["clouds"]["all"])
        country_code_list.append(response_json["sys"]["country"])
        date_list.append(response_json["dt"])
        humidity_list.append(response_json["main"]["humidity"])
        lat_list.append(response_json["coord"]["lat"])
        lng_list.append(response_json["coord"]["lon"])
        max_temp_list.append(response_json['main']['temp_max'])
        wind_speed_list.append(response_json["wind"]["speed"])

        if index > 49:
            index = 1
            set_count += 1 
        else:
            index += 1
        print(f"Processing Record #{index}, city name: {city}, set #{set_count}")
    except KeyError:
        print("Oops, that key doesn't exist.")

print("-----------------------------")
print("All Finished!")
print("-----------------------------")


# In[25]:


print(f"The first response is {json.dumps(response, indent=2)}.")


# In[26]:


print(len(city_list))
print(len(cloud_list))
print(len(country_code_list))
print(len(date_list))
print(len(humidity_list))
print(len(lat_list))
print(len(lng_list))
print(len(max_temp_list))
print(len(date_list))
print(len(wind_speed_list))


# In[57]:





# In[59]:


# Create a pandas dataframe using data retrieved
weather_df = pd.DataFrame({ 
                "City" : city_list,
                "Cloudiness" : cloud_list,
                "Country" : country_code_list,
                "Date" : date_list,
                "Humidity" : humidity_list,
                "Lat" : lat_list,
                "Lng" : lng_list,
                "Max Temp" : max_temp_list,
                "Wind Speed" : wind_speed_list
})


# In[58]:


weather_df.to_csv("output/cities_final.csv", index = False)


# In[60]:


weather_df


# In[61]:


weather_df.describe()


# In[64]:


# Create scatter plot
plt.scatter(weather_df["Lat"], weather_df["Max Temp"], facecolor = "red", edgecolor = "black")

# Set title, label and grid
plt.title("City Latitude vs. Maximum Temperature")
plt.xlabel("Laitude")
plt.ylabel("Maximum Temperature")
plt.grid(linewidth=.5, alpha = 1)

# Save as .pngs
plt.savefig("images/City Latitude vs Maximum Temperature.png")
print("The lower the latitude the more likely temperature is to drop")


# In[65]:


# Create scatter plot
plt.scatter(weather_df["Lat"], weather_df["Humidity"], facecolor = "yellow", edgecolor = "black")

# Set title, label and grid
plt.title("City Latitude vs. Humidity")
plt.xlabel("Laitude")
plt.ylabel("Humidity")
plt.grid(linewidth=.5, alpha = 1)

# Save as .pngs
plt.savefig("images/City Latitude vs. Humidity.png")
print("The higher latitude the higher the likelyhood of there being high humidity")


# In[42]:


# Create scatter plot
plt.scatter(weather_df["Lat"], weather_df["Cloudiness"], facecolor = "skyblue", edgecolor = "black")

# Set title, label and grid
plt.title("City Latitude vs. Cloudiness")
plt.xlabel("Laitude")
plt.ylabel("Cloudiness")
plt.grid(linewidth=.5, alpha = 1)

# Save as .pngs
plt.savefig("images/City Latitude vs. Cloudiness.png")
print("There doesn't appear to be much of a correlation betweent the two")


# In[43]:


# Create scatter plot
plt.scatter(weather_df["Lat"], weather_df["Wind Speed"], facecolor = "purple", edgecolor = "black")

# Set title, label and grid
plt.title("City Latitude vs. Wind Speed")
plt.xlabel("Laitude")
plt.ylabel("Wind Speed")
plt.grid(linewidth=.5, alpha = 1)

# Save as .pngs
plt.savefig("images/City Latitude vs. Wind Speed.png")
print("There are slightly higher wind speeds at higher latitudes")


# In[44]:


from datetime import datetime
ts = int("1611860897")
print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d'))


# In[45]:


weather_df['date'] = pd.to_datetime(weather_df['Date'],unit='s')


# In[46]:


weather_df


# In[47]:


#Sepperate northern and southern by latitude
Northern_Hem = weather_df[weather_df.Lat >= 0]
Southern_Hem = weather_df[weather_df.Lat <= 0]


# In[48]:


from sklearn.linear_model import LinearRegression


# In[77]:


#Create variables for northern and southern , temperature,humidity,coudiness and wind speed.
Lat_N = Northern_Hem.iloc[:, 5].values.reshape(-1, 1)  
Lat_S = Southern_Hem.iloc[:, 5].values.reshape(-1, 1)  

Temp_N = Northern_Hem.iloc[:, 7].values.reshape(-1, 1) 
Temp_S = Southern_Hem.iloc[:, 7].values.reshape(-1, 1) 

Humidity_N = Northern_Hem.iloc[:, 4].values.reshape(-1, 1) 
Humidity_S = Southern_Hem.iloc[:, 4].values.reshape(-1, 1)  

Cloudiness_N = Northern_Hem.iloc[:, 1].values.reshape(-1, 1)  
Cloudiness_S = Southern_Hem.iloc[:, 1].values.reshape(-1, 1) 

Wind_Speed_N = Northern_Hem.iloc[:, 8].values.reshape(-1, 1)  
Wind_Speed_S = Southern_Hem.iloc[:, 8].values.reshape(-1, 1)  

#Create predictor line
linear_regressor = LinearRegression() 
linear_regressor.fit(Lat_N, Temp_N)  
Temp_N_Predict = linear_regressor.predict(Lat_N)  


#Create plot and set labels, title and grid
plt.scatter(Lat_N, Temp_N, facecolor = "red", edgecolor = "black")
plt.title("Northern Hemisphere - Max Temp vs. Latitude Linear Regression")
plt.xlabel("Latitude")
plt.ylabel("Max Temp (F)")
plt.plot(Lat_N, Temp_N_Predict, color='blue')
plt.grid(linewidth=.5, alpha = 1)
plt.show()

plt.savefig("images/Northern Hemisphere - Max Temp vs. Latitude Linear Regression.png")


# In[78]:


#Create predictor line
linear_regressor = LinearRegression() 
linear_regressor.fit(Lat_S, Temp_S)  
Temp_S_Predict = linear_regressor.predict(Lat_S) 


#Create plot and set labels, title and grid
plt.scatter(Lat_S, Temp_S, facecolor = "red", edgecolor = "black")
plt.title("Southern Hemisphere - Max Temp vs. Latitude Linear Regression")
plt.xlabel("Latitude")
plt.ylabel("Max Temp (F)")
plt.plot(Lat_S, Temp_S_Predict, color='red')
plt.grid(linewidth=.5, alpha = 1)
plt.show()

plt.savefig("images/Southern Hemisphere - Max Temp vs. Latitude Linear Regression.png")


# In[79]:


#Create predictor line
linear_regressor = LinearRegression() 
linear_regressor.fit(Lat_S, Humidity_S)
Humidity_S_Predict = linear_regressor.predict(Lat_S) 


#Create plot and set labels, title and grid
plt.scatter(Lat_S, Humidity_S, facecolor = "yellow", edgecolor = "black")
plt.title("Southern Hemisphere - Humidity vs. Latitude Linear Regression")
plt.xlabel("Latitude")
plt.ylabel("Humidity")
plt.plot(Lat_S, Humidity_S_Predict, color='red')
plt.grid(linewidth=.5, alpha = 1)
plt.show()

plt.savefig("images/Southern Hemisphere - Humidity vs. Latitude Linear Regression.png")


# In[80]:


#Create predictor line
linear_regressor = LinearRegression() 
linear_regressor.fit(Lat_N, Humidity_N)  
Humidity_N_Predict = linear_regressor.predict(Lat_N)  


#Create plot and set labels, title and grid
plt.scatter(Lat_N, Humidity_N, facecolor = "yellow", edgecolor = "black")
plt.title("Northern Hemisphere - Humidity vs. Latitude Linear Regression")
plt.xlabel("Latitude")
plt.ylabel("Humidity")
plt.plot(Lat_N, Humidity_N_Predict, color='blue')
plt.grid(linewidth=.5, alpha = 1)
plt.show()

plt.savefig("images/Northern Hemisphere - Humidity vs. Latitude Linear Regression.png")


# In[81]:


#Create predictor line
linear_regressor = LinearRegression()  
linear_regressor.fit(Lat_S, Cloudiness_S) 
Cloudiness_S_Predict = linear_regressor.predict(Lat_S) 



#Create plot and set labels, title and grid
plt.scatter(Lat_S, Cloudiness_S, facecolor = "skyblue", edgecolor = "black")
plt.title("Southern Hemisphere - Cloudiness vs. Latitude Linear Regression")
plt.xlabel("Latitude")
plt.ylabel("Cloudiness")
plt.plot(Lat_S, Cloudiness_S_Predict, color='red')
plt.grid(linewidth=.5, alpha = 1)
plt.show()

plt.savefig("images/Southern Hemisphere - Cloudiness vs. Latitude Linear Regression.png")


# In[82]:


#Create predictor line
linear_regressor = LinearRegression()  
linear_regressor.fit(Lat_N, Cloudiness_N)  
Cloudiness_N_Predict = linear_regressor.predict(Lat_N)  

#Create plot and set labels, title and grid
plt.scatter(Lat_N, Cloudiness_N, facecolor = "skyblue", edgecolor = "black")
plt.title("Northern Hemisphere - Cloudiness vs. Latitude Linear Regression")
plt.xlabel("Latitude")
plt.ylabel("Cloudiness")
plt.plot(Lat_N, Cloudiness_N_Predict, color='blue')
plt.grid(linewidth=.5, alpha = 1)
plt.show()

plt.savefig("Northern Hemisphere - Cloudiness vs. Latitude Linear Regression.png")


# In[83]:


#Create predictor line
linear_regressor = LinearRegression()  
linear_regressor.fit(Lat_N, Wind_Speed_N)  
Wind_Speed_N_Predict = linear_regressor.predict(Lat_N)  

#Create plot and set labels, title and grid
plt.scatter(Lat_N, Wind_Speed_N, facecolor = "purple", edgecolor = "black")
plt.title("Northern Hemisphere - Wind_Speed vs. Latitude Linear Regression")
plt.xlabel("Latitude")
plt.ylabel("Wind_Speed")
plt.plot(Lat_N, Wind_Speed_N_Predict, color='blue')
plt.grid(linewidth=.5, alpha = 1)
plt.show()

plt.savefig("Northern Hemisphere - Wind_Speed vs. Latitude Linear Regression.png")


# In[84]:


#Create predictor line
linear_regressor = LinearRegression()  
linear_regressor.fit(Lat_S, Wind_Speed_S)  
Wind_Speed_S_Predict = linear_regressor.predict(Lat_S)  

#Create plot and set labels, title and grid
plt.scatter(Lat_S, Wind_Speed_S, facecolor = "purple", edgecolor = "black")
plt.title("Southern Hemisphere - Wind_Speed vs. Latitude Linear Regression")
plt.xlabel("Latitude")
plt.ylabel("Wind_Speed")
plt.plot(Lat_S, Wind_Speed_S_Predict, color='red')
plt.grid(linewidth=.5, alpha = 1)
plt.show()

plt.savefig("Southern Hemisphere - Wind_Speed vs. Latitude Linear Regression.png")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




