#imports for streamlit and mapping
import streamlit as st
import pandas as pd
import numpy as np
import glob
import statsmodels.api as sm
import pydeck as pdk
import os 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 


#imports for the plots
from PIL import Image
import graphviz as graphviz
import altair as alt
import colorlover as cl
from IPython.display import HTML
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as ply
import plotly.graph_objs as go
from plotly.tools import make_subplots

#This code contains maps and plots with Streamlit
#The code is dived in two parts, the first parts contains a visual representation of the trips,
#maps and costumes engagement in an average hours format
#The second part of the code includes the plots

# read data from csv for the maps
df = pd.read_csv('full_data.csv')
 
#Getting data from the csv files for the plots

chosen_colors=cl.scales['5']['qual'][np.random.choice(list(cl.scales['5']['qual'].keys()))]
station = pd.read_csv('station11.csv')
trip = pd.read_csv('trip01230.csv')
trip.head()

#trip dates for the plots
trip.start_date=pd.to_datetime(trip.start_date,infer_datetime_format=True)
trip.end_date=pd.to_datetime(trip.end_date,infer_datetime_format=True)

import re              
import datetime        
                  # importing date time for the datepart

def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Date', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


    add_datepart(trip, 'start_date', drop=False, time=True)
    add_datepart(trip, 'end_date', drop=False, time=True)
    trip.head()

# header for the first part (mapping)
st.title("Bagboard –  The sustainable trips from our customers")
st.subheader("Visual representation of all the trips made in UK")

# arc plot map (1)

    # this map shows all the trips done from the user
    #it take the starting point and the end point, it does not show the route but just the distance from start to end
    #represented with the arc

df_id = df[['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']]
midpoint = (np.average(df_id["end_latitude"]),
            np.average(df_id["end_longitude"]))

#mapbox used for the arc plot map
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=pdk.ViewState(
        latitude=midpoint[0],
        longitude=midpoint[1],
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            "ArcLayer",
            data=df_id,
            get_source_position="[start_longitude, start_latitude]",
            get_target_position="[end_longitude, end_latitude]",
            get_source_color=[0, 30, 87, 160],
            get_target_color=[0, 30, 190, 160]

        )
    ]
))

#This section of the first part contains a map that shows the costumer engagement
 # Data is taken from the csv files
 # What the code does, is that it takes all the costumer engagement from all the days they have been active
 #and shows the activity in each hour


#Mapping
my_dataset = 'journey_locations.csv' # for this plot we take the data from the journey locations
DATE_TIME = "date/time"

st.title("Costumer Engagement")
st.markdown(
"""
This map represents the average of costumer engagement during the 24 hours, in hour-format.
""")           

#this can help to decide what time of the day is the most appropriate to launch an addvertisment and where

@st.cache(persist=True) #using cache to load faster 
def load_data(nrows):
    data = pd.read_csv(my_dataset, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
    return data

data = load_data(100000)

hour = st.slider("Hour to look at", 0, 23)  # facing problems to make  a range of hours/ but it runs when

data = data[data[DATE_TIME].dt.hour == hour]

#time bar where you can decide at what hour you want 
st.subheader("Geo data between %i:00 and %i:00" % (hour, (hour + 1) % 24)) 
midpoint = (np.average(data["lat"]), np.average(data["lon"]))


#the map shows hexagon layer representing the activity during the selected hours.
st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=data,
            get_position=["lon", "lat"],
            radius=100,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
    ],
))

#############################################################################################################################################

#The following section contains the map that is that shows the costumeR activity dynamics in UK
#you can select which type of data you want to display
#the types of data show the increase ratio (dummy data from the csv), costumer activy, over all activities
#the data is represented with circles, colors of circles corresponding to the data  shown 

#############################################################################################################################################


mapbox_access_token = 'pk.eyJ1IjoiY29qYWNrIiwiYSI6IkRTNjV1T2MifQ.EWzL4Qk-VvQoaeJBfE6VSA'
px.set_mapbox_access_token(mapbox_access_token)

det = st.checkbox('Press for detailed explanation',value=True)

### LOAD DATA ### activities

def load_data():
    import os 
    import time
    import datetime

    try:
        dummyt = pd.to_datetime(time.ctime(os.path.getmtime('dummy_casesx2.csv')))
    except:
        dummyt = pd.to_datetime

    if (pd.to_datetime(datetime.datetime.now())-dummyt).days > 0:
        #reload all data
        data_load_state = st.text('Loading data and running initial analyses...')
        #get file list
        fi = glob.glob("./*")
        

    else:
        #use stored results:
        LKx = pd.read_csv('LKposi.csv',index_col=0)
        dummy_casesx2 = pd.read_csv('dummy_casesx2.csv',index_col=0)
        dummy_casesx2.columns = pd.to_datetime(dummy_casesx2.columns)
        dummy_increase = pd.read_csv('dummy_increase.csv',index_col=0)
        dummy_increase.columns = pd.to_datetime(dummy_increase.columns)
        dummy_frowfac = pd.read_csv('dummy_frowfac.csv',index_col=0)
        dummy_frowfac.columns = pd.to_datetime(dummy_frowfac.columns)
        dummy_double = pd.read_csv('dummy_double.csv',index_col=0)
        dummy_double.columns = pd.to_datetime(dummy_double.columns)

    return [pd.concat([LKx,dummy_casesx2],axis=1),dummy_increase,dummy_frowfac,dummy_double,LKx]



[data_case,data_increase,data_frowfac,data_double,LKx] = load_data()




### MAP ### Activities

if det:
  
    st.subheader('Bagboard costumer activity dynamics in UK')
  
data_sel = st.selectbox('Select Data',['Activities','Activities per 100000 capita','Activities increase','Activities increase per 100000 capita','Increase ratio'],4)
if data_sel=='Activities':
    data_cases = data_case
elif data_sel=='Activities per 100000 capita':
    data_cases = pd.concat([LKx,data_case.iloc[:,5:].div(LKx.id1,axis=0)*100000.],axis=1)
elif data_sel=='Activities increase':
    data_cases = pd.concat([LKx,data_increase],axis=1)
elif data_sel=='Activities increase per 100000 capita':
    data_cases = pd.concat([LKx,data_increase.div(LKx.id1,axis=0)*100000.],axis=1)
elif data_sel=='Increase ratio':
    data_cases = pd.concat([LKx,data_frowfac],axis=1)




#helper function

#jiter cases
dlat = 0.6#np.exp(np.log(data_cases.lat.sort_values().diff()).median())*50.
dlon = 0.6#np.exp(np.log(data_cases.lon.sort_values().diff()).median())*50.

def jiter_data(data,co):
    firstitem = True
    for i in data.index:
        n = int(data_cases.loc[i,co])
        c = pd.DataFrame((np.random.randn(2*n).reshape((n,2))*0.5)*np.array([dlat,dlon])+data.loc[i,['lat','lon']].values,columns=['lat','lon'])
        if firstitem:
            dummyj = c
            firstitem = False
        else:
            dummyj = pd.concat([dummyj,c])
    dummyj = dummyj.reset_index()
    return dummyj

if (data_sel == 'Activities') | (data_sel=='Activities per 100000 capita'):
    hmplot = st.checkbox('Show heatmap')
else:
    hmplot = False

di = st.slider('day', 5, len(data_cases.columns)-1, len(data_cases.columns)-1)
datex = data_cases.columns[di]
    
if ((data_sel=='Activities') | (data_sel=='Activities per 100000 capita')) & (hmplot==True):
    #show map with hexagon heatmap
    
    # Define a layer to display on a map
    layer = pydeck.Layer(
       'HexagonLayer',
       jiter_data(data_cases,datex)[['lon', 'lat']],
       get_position=['lon', 'lat'],
       auto_highlight=True,
       elevation_scale=100,
       pickable=True,
       colorRange=[[69,2,86],[59,28,140],[33,144,141],[90,200,101],[249,231,33]],
       elevation_range=[0,3000],
       elevationDomain=[0,12],
       extruded=True,
       coverage=10)
    
    # Set the viewport location
    view_state = pydeck.ViewState(
        longitude=8.815,
        latitude=51.155323,
        zoom=5,
        pitch=25.5,
        bearing=0.)
    
    st.subheader('User activities on ' + str(datex.date()))
    st.pydeck_chart(
        pydeck.Deck(layers=[layer], initial_view_state=view_state, mapbox_key='pk.eyJ1IjoiY29qYWNrIiwiYSI6IkRTNjV1T2MifQ.EWzL4Qk-VvQoaeJBfE6VSA')
        )

elif (data_sel=='Activities') & (hmplot==False):
    data_cases1 = pd.concat([data_cases[['lat','lon',datex,'Location','id1']],data_increase[datex],data_frowfac[datex]],axis=1)
    data_cases1.columns = ['lat','lon','cases','Location','capita','increase','growth']
    fig = px.scatter_mapbox(data_cases1, lat="lat", lon="lon",  size='cases', color=np.log10(data_cases1.cases),
                 
                  color_continuous_scale=px.colors.sequential.Cividis, size_max=24, zoom=10, height=600)
    fig.update_layout(coloraxis_colorbar=dict(
        title="Activities",
        tickvals=[0,1,2,3],
        ticktext=['1' , '10', '100', '1000'],
        ))
    st.subheader('User activities on ' + str(datex.date()))
    st.plotly_chart(fig)

elif (data_sel=='Activities per 100000 capita') & (hmplot==False):
    data_cases1 = pd.concat([data_cases[['lat','lon',datex,'Location','id1']],data_increase[datex],data_frowfac[datex]],axis=1)
    data_cases1.columns = ['lat','lon','cases','Location','capita','increase','growth']
    fig = px.scatter_mapbox(data_cases1, lat="lat", lon="lon",  size='cases', color='cases',
                  color_continuous_scale=px.colors.sequential.Cividis, size_max=24, zoom=10, height=600)
    if det:
        st.subheader('to be edited...' + str(datex.date()))
        st.markdown('to be edited...')
    else:
        st.subheader('User activities on ' + str(datex.date()))
        st.markdown('Color and size give the number of cases per 100000 inhabitants.')
    st.plotly_chart(fig)

else:
    data_cases1 = pd.concat([data_case[['lat','lon',datex,'Location','id1']],data_case[datex].div(LKx.id1,axis=0)*100000.,data_increase[datex],data_increase[datex].div(LKx.id1,axis=0)*100000.,data_double[datex],data_frowfac[datex]],axis=1)
    data_cases1.columns = ['lat','lon','cases','Location','capita','cases per 100000','increase','increase per 100000','days to double','growth']
    #data_cases1 = data_cases1.dropna()
    if (data_sel=='Activities increase'):
        fig = px.scatter_mapbox(data_cases1, lat="lat", lon="lon",  size='cases per 100000', color='increase',
                  color_continuous_scale=px.colors.diverging.Portland, size_max=24, zoom=10, height=600)
        if det:
            st.subheader('to be edited...' + str(datex.date()))
            st.markdown('to be edited...')
        else:
            st.subheader('User activities on ' + str(datex.date()))
            st.markdown('Color gives total Activities. Size gives Activities per capita.')
        st.plotly_chart(fig)
    elif (data_sel=='Activities increase per 100000 capita'):
        fig = px.scatter_mapbox(data_cases1, lat="lat", lon="lon",  size='cases per 100000', color='increase per 100000',
                  color_continuous_scale=px.colors.diverging.Portland, size_max=24, zoom=5, height=600)
        if det:
            st.subheader('to be edited... ' + str(datex.date()))
            st.markdown('to be edited...')
        else:
            st.subheader('User activities  increase per 100000 on ' + str(datex.date()))
            st.markdown('Color gives the increase of Activities. Size gives Activities per capita.')
        st.plotly_chart(fig)
    
    elif (data_sel=='Increase ratio'):
        fig = px.scatter_mapbox(data_cases1, lat="lat", lon="lon",  size='cases per 100000', color='growth',
                  color_continuous_scale=px.colors.diverging.Portland, size_max=24, zoom=10, height=600)
        st.subheader('User activities growth on ' + str(datex.date()))
        st.markdown('Color gives growth ratio to last case increase. Size gives cases per capita.')
        st.plotly_chart(fig)





############################ Plots ##########################################################################################################

# header
st.title("Bagboard –  The sustainable trips from our customers")
st.subheader("Visual representation of all the trips made in UK")


#The second part uses mainly dummy data for the plots, 
#There is a representation of the locations(stations) and number of activities in those areas


#reading data from the file
df = pd.read_csv('station11.csv')

#grouping the data by stations(name of locations)
citygroup=station.groupby(['city'])
temp_df1=citygroup['id'].count().reset_index().sort_values(by='id', ascending=False)
temp_df2=citygroup['dock_count'].sum().reset_index().sort_values(by='dock_count', ascending=False)

#first plot will show the y axis-(representing the no. of activites) and x axis( showing the city(station areas))
trace1 = go.Bar(
    x=temp_df1.city,
    y=temp_df1.id,
    name='No. of stations',
    text=temp_df1.id,
    textposition='outside',
    marker=dict(color=chosen_colors[0]
        
    )
)

data=[trace1]

layout = go.Layout(
    title='No. of activities in different zones',
    xaxis=dict(
        title='City'
    ),
    yaxis=dict(
        title='No. of activities in different zones'
    ),
)

figure = go.Figure(data=data, layout=layout)

st.plotly_chart(figure)



########## “clustered” plotting with two variables, in this case we are taking the nr. of station and the number of the trips 

trace2 = go.Bar(
    x=temp_df2.city,
    y=temp_df2.dock_count,
    name='No. of trips',
    text=temp_df2.dock_count,
    textposition='auto',
    marker=dict(
        color=chosen_colors[2]
    )
)

data=[trace1, trace2]

figure = go.Figure(data=data, layout=layout)

figure['layout'].update(dict(title='No. of trips'), barmode='group')

st.plotly_chart(figure)


#######Trips

trip.start_date=pd.to_datetime(trip.start_date,infer_datetime_format=True)
trip.end_date=pd.to_datetime(trip.end_date,infer_datetime_format=True)

import re
import datetime
def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)
    
add_datepart(trip, 'start_date', drop=False, time=True)
add_datepart(trip, 'end_date', drop=False, time=True)


#Duration distribution plot
 #showing how many trips in relation with the time taken from the trips to be completed

trip['duration_min']=trip.duration/60
trace1 = go.Histogram(
    x=trip[trip.duration_min<60].duration_min, #To remove outliers
    marker=dict(
        color=chosen_colors[0]
    )    
)

data=[trace1]

layout = go.Layout(
    title='Distribution of  trips duration',
    xaxis=dict(
        title='Trip Duration (minutes)'
    ),
    yaxis=dict(
        title='Count'
    ),
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

st.plotly_chart(figure)


#Spliting the histogram by campaign type
#the code will search for the type of campaigns in the csv file and will display the count of the campains
#in relation to the time needed for the campaign to be completed

data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    data.append(
        go.Histogram(
            x=trip[(trip.campaign_type==trace_names[i]) & (trip.duration_min<60)].duration_min,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            ),
            opacity=0.5
        )
    )

layout = go.Layout(
    title='Distribution of trip duration ',
    barmode='overlay',
    xaxis=dict(
        title='Trip Duration (minutes)'
    ),
    yaxis=dict(
        title='Count'
    ),
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

st.plotly_chart(figure)



#This plot will represent the number of trips undertaken in each month


trip_count_by_month=trip.groupby(['start_Year','start_Month'])['id'].count().reset_index()

trace1 = go.Bar(
    x=trip_count_by_month.start_Month.astype(str)+'-'+trip_count_by_month.start_Year.astype(str),
    y=trip_count_by_month.id,
    name='No. of trips',
    marker=dict(
        color=chosen_colors[0]
    )
)

data=[trace1]

layout = go.Layout(
    title='No. of trips by month',
    xaxis=dict(
        title='Month'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
)

figure = go.Figure(data=data, layout=layout)

st.plotly_chart(figure)




#This plot will compare two campains by representing the number of trips undertaken in each month


trip_count_by_month_sub=trip.groupby(['start_Year','start_Month', 'campaign_type'])['id'].count().reset_index()

data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    temp_df=trip_count_by_month_sub[(trip_count_by_month_sub.campaign_type==trace_names[i])]
    data.append(
        go.Bar(
            x=temp_df.start_Month.astype(str)+'-'+temp_df.start_Year.astype(str),
            y=temp_df.id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='No. of trips per month',
    xaxis=dict(
        title='Month'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    barmode='stack'
)

figure = go.Figure(data=data, layout=layout)

st.plotly_chart(figure)



#Popular days of the week
#this code will take the data from the trips and will divide the trips by the days of the week and the weekends
#at the ende the code will show for each month if they had a higher activity during the week or the weekends
trip['start_date_dt']=[i.date() for i in trip.start_date]
trip['end_date_dt']=[i.date() for i in trip.end_date]

trip_count_by_date=trip.groupby(['start_date_dt'])['id'].count().reset_index()
trip_count_by_date['day_of_week']=[i.weekday() for i in trip_count_by_date.start_date_dt]
trip_count_by_date['is_weekend'] = (trip_count_by_date.day_of_week>4)*1
data=[]

trace_names=['Weekday', 'Weekend']

for i in range(2):
    data.append(
        go.Bar(
            x=trip_count_by_date[(trip_count_by_date.is_weekend==i) & (trip_count_by_date.start_date_dt<datetime.date(2014, 1, 1))].start_date_dt,
            y=trip_count_by_date[(trip_count_by_date.is_weekend==i)  & (trip_count_by_date.start_date_dt<datetime.date(2014, 1, 1))].id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )


layout = go.Layout(
    title='No. of trips per day ',
    xaxis=dict(
        title='Date'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
)

figure = go.Figure(data=data, layout=layout)

st.plotly_chart(figure)



