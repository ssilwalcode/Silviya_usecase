# pip install necessary packages
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from math import sin, cos, sqrt, atan2, radians
from PIL import Image
from PIL import ImageDraw, ImageFont

#read station csv
station = pd.read_csv('datasets/london_cycle_stations - london_cycle_stations.csv')

# Possibility to see the cleaned data
def clean_extract(df, station):
    
    df = df.drop_duplicates()
    #cleaning the datetime columns
    date_cols = ['start_date', 'end_date']
    df = df[df[date_cols[0]].str.contains(r'^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\sUTC)$', regex=True)]
    df = df[df['start_station_name'].isin(station['name'])]

    #changing the type to datetime
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], utc=True)
    #cleaning the values
    df = df[df['duration'].str.contains(r'^[+-]?\d*\.?\d+$', regex=True)]
    df['duration'] = df['duration'].astype(float)
    df['duration'] = df['duration'].astype(int)
    #converting to minutes
    df['duration'] = df['duration']  / 60

    #dropping rows with null values
    df = df[df['start_station_id'].notnull()]
    df['start_station_id'] = df['start_station_id'].astype(int)
    
    #checking and removing  end station name for error values
    df = df[df['end_station_id'].str.contains(r'^[+-]?\d*\.?\d+$', regex=True)]
    df['end_station_id'] = df['end_station_id'].astype(int)
    return df

#1 Top 10 of most/less popular stations regarding the weekdays.
def popular_station(df, top_k=10):
    df['day_of_week'] = df['start_date'].dt.day_name()
    day = df['start_station_name'].value_counts().to_dict()
    day = [(k, v) for k, v in sorted(day.items(), key=lambda item: item[1], reverse=True)]
    # most_popular = day[:top_k]
    # least_popular = day[-top_k:]
    return day

# What is the distribution of bike rental duration?
def duration(df, k=120):
    less = df[df['duration']<k]['duration'].array
    return less

# Where are the rental stations that have the most turnover rate?
def turnover_rate(extract_df, station_df, top_k=10):
    total_rentals = extract_df.groupby('start_station_id')['bike_id'].count() + extract_df.groupby('end_station_id')['bike_id'].count()

    # number of unique rentals for each rental station
    unique_rentals = extract_df.groupby('start_station_id')['bike_id'].nunique() + extract_df.groupby('end_station_id')['bike_id'].nunique()

    # turnover rate for each rental station ??
    turnover_rate = unique_rentals / total_rentals

    # Sorting the rental stations by their turnover rate in descending order
    top_turnover_stations = turnover_rate.sort_values(ascending=False).head(top_k)
    
    turnovers = list(top_turnover_stations.to_dict().keys())
    
    result = station_df[station_df['id'].isin(turnovers)][['id', 'name']]
    
    return result
# (OPTIONAL) Convert graphs into dashboard so it’s possible to filter

# What is the average distance between two stations ?
def distance(lat1, lon1, lat2, lon2):
    # approximate radius of Earth in km
    R = 6373.0
    # convert degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    # differences between the latitudes and longitudes
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # apply the Haversine formula to calculate the distance between the two points
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance

def station_distance(df, station1, station2):
    station = df[df['name'].isin([station1, station2])][['latitude', 'longitude']]
    s1 = df[df['name']==station1][['latitude', 'longitude']].to_records().tolist()[0]
    s2 = df[df['name']==station2][['latitude', 'longitude']].to_records().tolist()[0]
    dis = distance(s1[1], s1[2], s2[1], s2[2])
    # Open an Image
    img = Image.open('route.jpg')

    # draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(img)
    font = ImageFont.truetype('LucidaSansRegular.ttf', 20)

    # Adding text to image
    I1.text((60,50),station1, font = font, fill=(255, 0, 0))

    I1.text((400,270),station2, font = font,fill=(255, 0, 0))

    I1.text((250,415),f'{dis:.2f} KM', font = font, fill=(255, 0, 0))
    newsize = (360, 360)
    # img = img.resize(newsize)
    st.image(img, caption=f'Distance between {station1} and {station2} is {dis:.2f} KM')
    return dis

#What schedules to avoid to be on time at my job? (from home to work)

def avoid(extract_df, my_station='Green Street, Mayfair'):
    busy = extract_df[extract_df['start_station_name']==my_station][['start_date']]
    busy['hours'] = busy.start_date.dt.hour
    return busy

#calculating overall average distance of the stations in london
lats = station[['name', 'latitude', 'longitude']].to_records().tolist()
total_distance = 0
for idx, (_, name, lat1, long1) in enumerate(lats):
    for jdx, (_, name, lat2, long2) in enumerate(lats):
        if idx != jdx:
            dist = distance(lat1, long1, lat2, long2)
            total_distance += dist

avg_distance = total_distance/ (len(lats)* (len(lats) - 1))


#streamlit

if 'map' not in st.session_state:
    st.session_state['map'] = None

if 'table' not in st.session_state:
    st.session_state['table'] = None

if 'dist' not in st.session_state:
    st.session_state['dist'] = None

if 'topK' not in st.session_state:
    st.session_state['topK'] = 120

if 'duration_dist' not in st.session_state:
    st.session_state['duration_dist'] = None

if 'download' not in st.session_state:
    st.session_state['download'] = False

if 'turnover' not in st.session_state:
    st.session_state['turnover'] = None

# m = st.markdown("""
# <style>
# div.stButton > button:first-child {
#     background-color: rgb(209, 237, 242);
# }
# </style>""", unsafe_allow_html=True)

new_title = '<p style="font-family:sans-serif; color:Black; font-size: 42px; font-weight:bold">London Bicycle Fleet Data</p>'
st.markdown(new_title, unsafe_allow_html=True)

#upload extract csv
data_file = st.file_uploader("Upload your csv here", type={"csv"})
if data_file is not None:
    bike_df = pd.read_csv(data_file)
    if st.button('Clean data ▶️'):
        clean = clean_extract(bike_df, station)
        st.write('Preview of clean data')
        st.write(clean.head())
        st.session_state['clean'] = clean
        st.session_state['download'] = True

 #download the clean data      
if st.session_state['download']:
    clean = st.session_state['clean']
    st.download_button('Download cleaned CSV 📥', clean.to_csv().encode('utf-8'), 'clean_extract_data.csv')

#select stations to find the distance -- Haversine distance
s1 = st.selectbox('Select start station', station['name'].unique())
s2 = st.selectbox('Select end station', station['name'].unique())

if s1 is not None and s2 is not None:
    if st.button('Show distance 🚴'):
        #this is the overall diistance
        st.write(f'The overall average distance between the stations in london is {avg_distance:.2f} KM')

        #this is the distance between the two stations selected by user -- Haversine distance
        station_distance(station, s1, s2)
        # bike_df = pd.read_csv(data_file)
        clean = st.session_state['clean']
        busy_title = f'<p style="font-family:sans-serif; color:Black; font-size: 22px; font-weight:bold">Busiest times in {s1}</p>'
        st.markdown(busy_title, unsafe_allow_html=True)
        busy = avoid(clean, s1)
        fig, p = plt.subplots()
        p = busy.hours.plot(kind='hist', bins=24, color='orange')
        p.set_xlabel('Time of day')
        p.set_ylabel('Frequency of bikes rented')
        index_of_bar_to_label = np.argmax(np.histogram(busy.hours, bins=24)[0])

        p.patches[index_of_bar_to_label].set_color('r')
        st.pyplot(fig)

# Map of the stations
if st.button('Global data visualization 📊'):
    clean = st.session_state['clean']
    m = popular_station(clean)
    data = []
    for k,v in m:
        try:
            s1 = station[station['name']==k][['latitude', 'longitude']].to_records().tolist()[0]
            
            data.append([k,v,s1[1],s1[2]])
        except:
            print(len(s1),k)

    df = pd.DataFrame(data, columns=['name','popularity','latitude','longitude'])
    color_scale = [(0, 'orange'), (1,'red')]

    fig = px.scatter_mapbox(df, 
                            lat="latitude", 
                            lon="longitude", 
                            hover_name= "name", 
                            color="popularity",
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            size="popularity",
                            zoom=10, 
                            height=800,
                            width=800)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.session_state['map'] = fig
    pop={'Top 10 most popular stations': [k for k, v in m[:10]], 'Top 10 least popular stations': [k for k, v in m[-10:]]}
    pop=pd.DataFrame(pop)
    st.session_state['table'] = pop
    
    
    # st.session_state['topK'] = 120
    less = duration(clean, 30)
    st.session_state['duration_dist'] = less

    top_turn = turnover_rate(clean, station, 15)
    st.session_state['turnover'] = top_turn

if st.session_state['map'] is not None:
    map_title = '<p style="font-family:sans-serif; color:Black; font-size: 22px; font-weight:bold">Map of stations in London</p>'
    st.markdown(map_title, unsafe_allow_html=True)
    # st.write('Map of stations in London')
    st.plotly_chart(st.session_state['map'],  use_container_width=True)

if st.session_state['table'] is not None:
    popular_title = '<p style="font-family:sans-serif; color:Black; font-size: 22px; font-weight:bold">Top 10 most and least popular stations</p>'
    st.markdown(popular_title, unsafe_allow_html=True)
    st.table(st.session_state['table'].assign(hack='').set_index('hack'))

if st.session_state['duration_dist'] is not None:
    clean = st.session_state['clean']
    slider_title = '<p style="font-family:sans-serif; color:Black; font-size: 22px; font-weight:bold">Distribution of Bikes rented (in minutes)</p>'
    st.markdown(slider_title, unsafe_allow_html=True)
    k = st.slider('Select Maximum Duration', 0,150,120,2)
    st.session_state['topK'] = k
    less = duration(clean, st.session_state['topK'])
    st.session_state['duration_dist'] = less
    fig = px.histogram(st.session_state['duration_dist'], nbins=50, range_x=(0,st.session_state['topK']),
                   labels = {'value':'Duration (minutes)', 'count': 'Frequency'}, 
                   )
    fig.update_layout(showlegend=False)


    # st.write('Map of stations in London')
    st.plotly_chart(fig,  use_container_width=True)

if st.session_state['turnover'] is not None:
    turnover_title = '<p style="font-family:sans-serif; color:Black; font-size: 22px; font-weight:bold">Rental stations with most turnover rate</p>'
    st.markdown(turnover_title, unsafe_allow_html=True)
    # st.write(f'Rental stations with most turnover rate ')
    st.table(st.session_state['turnover'].assign(hack='').set_index('hack'))