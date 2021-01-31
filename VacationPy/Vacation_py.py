#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import tools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import time
from config import api_key
from config import gkey
import gmaps
import os
import json


# In[2]:


#read csv
weather_df = pd.read_csv("../Weather.py/output/cities_final.csv")
weather_df


# In[3]:


gmaps.configure(api_key=gkey)

# Store latitude and longitude in locations abd humidity in a humidity variable
locations = weather_df[["Lat", "Lng"]]
humidity = weather_df["Humidity"]


# In[4]:


# Plot Heatmap
fig = gmaps.figure(center=(46.0, -5.0), zoom_level=1)
max_intensity = np.max(humidity)

# Create heat layer
heat_layer = gmaps.heatmap_layer(locations, weights = humidity, dissipating=False, max_intensity=100, point_radius=2)

# Add heat layer
fig.add_layer(heat_layer)

#show fig
fig


# In[5]:


#Find cities using given parameters
city_df = weather_df.loc[(weather_df["Max Temp"] < 80)                                 & (weather_df["Max Temp"] > 70)                                 & (weather_df["Wind Speed"] < 10)                                 & (weather_df["Cloudiness"] == 0)].dropna() 


# In[6]:


#Add Hotel Name column
city_df['Hotel Name'] = ""
city_df


# In[7]:


#Show only the values we need
hotel_df = city_df[['Hotel Name','City','Country','Lat','Lng']]

#ser base url and parameters
base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
params = {"type" : "lodging",
          "keyword" : "hotel",
          "radius" : 5000,
          "key" : gkey}


# In[ ]:





# In[8]:


#Create for loop
for index, row in hotel_df.iterrows(): 
    #grab the longitude, latitude, city and country from city_df
    lat = row["Lat"]
    lng = row["Lng"]
    city = row["City"]
    Country = row["Country"]
    
    #Add location to the parameter dictionary
    params["location"] = f"{lat},{lng}"
    
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"  
    
    response = requests.get(base_url,params=params).json()
    
    #create a variable with the results
    results = response['results']
    
    try:
        #Add Hotel Name to hotel_df
        hotel_df.loc[index,"Hotel Name"] = results[0]['name']
        print(f"{results[0]['name']} is the closest hotel to {city}")

    except (KeyError, IndexError):
            print("Missing information")


# In[11]:


#drop the rows that are missing information
hotel_df = (hotel_df.drop(hotel_df.index[[1, 6]]))
hotel_df


# In[12]:


# NOTE: Do not change any of the code in this cell

# Using the template add the hotel marks to the heatmap
info_box_template = """
<dl>
<dt>Name</dt><dd>{Hotel Name}</dd>
<dt>City</dt><dd>{City}</dd>
<dt>Country</dt><dd>{Country}</dd>
</dl>
"""
# Store the DataFrame Row
# NOTE: be sure to update with your DataFrame name
hotel_info = [info_box_template.format(**row) for index, row in hotel_df.iterrows()]
locations = hotel_df[["Lat", "Lng"]]


# In[13]:


# Add marker layer and info box content ontop of heat map
markers = gmaps.marker_layer(locations, info_box_content = hotel_info)
fig.add_layer(markers)

# Display Map
fig

