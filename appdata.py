import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import colorlover as cl
from IPython.display import HTML
import plotly.offline as ply
import plotly.graph_objs as go
from plotly.tools import make_subplots

chosen_colors=cl.scales['5']['qual'][np.random.choice(list(cl.scales['5']['qual'].keys()))]
station = pd.read_csv('station11.csv')
trip = pd.read_csv('trip01230.csv')
trip.head()






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
    attr = ['Year', 'Month', 'Week', 'Day', 'Date', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


    add_datepart(trip, 'start_date', drop=False, time=True)
    add_datepart(trip, 'end_date', drop=False, time=True)
    trip.head()

# header
st.title("Bagboard –  The sustainable trips from our customers")
st.subheader("Visual representation of all the trips made in UK")

df = pd.read_csv('station11.csv')


citygroup=station.groupby(['city'])
temp_df1=citygroup['id'].count().reset_index().sort_values(by='id', ascending=False)
temp_df2=citygroup['dock_count'].sum().reset_index().sort_values(by='dock_count', ascending=False)

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



########## “clustered” column 


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


#Duration distribution
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


#Now let us split the histogram by campaign type

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




#Let us start with plotting the no. of trips by month.

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

#Campaign type differ by month


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



#trial
trip_count_by_date=trip.groupby(['start_date'])['id'].count().reset_index()

trace1 = go.Scatter(
    x=trip_count_by_date.start_date,
    y=trip_count_by_date.id,
    mode='lines',
    line=dict(
        color=chosen_colors[0]
    ),
    name='Daily'
)

data=[trace1]


layout = go.Layout(
    title='No. of trips per day ',
    xaxis=dict(
        title='Date',
        showgrid=False
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    hovermode='closest',
)

figure = go.Figure(data=data, layout=layout)


st.plotly_chart(figure)