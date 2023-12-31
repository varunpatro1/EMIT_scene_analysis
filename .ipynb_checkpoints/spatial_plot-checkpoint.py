import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point

def spatial_plt(df):
    geometry = [Point(xy) for xy in zip(df['lat'], df['lon'])]
    gdf = GeoDataFrame(df, geometry = geometry)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax = world.plot(figsize = (15,15)), marker = 'o', color = 'red', markersize = 3);


def metric_variance_plt(df, metric):
    title = 'Global Map of ' + metric + ' Variance'
    fig, ax = plt.subplots(figsize=(20,16), ncols = 1, nrows = 2, gridspec_kw = None)
    countries = gpd.read_file(  
    gpd.datasets.get_path("naturalearth_lowres"))

    for i in range(2):
        countries.plot(color="lightblue", ax = ax[i])
        
    # plot points
    df_metric = metric
    for i in range(2):
        df.plot.scatter(x="lat", y="lon", s = 5, c=df[df_metric], colormap="plasma", title= title, ax=ax[i])


def elev_zen_variance_plt(df, sampling_factors):
    #title_elev = 'Global Map of Elevation Variance based on' + sampling_factors
    #title_zen = 'Global Map of Zenith Angle Variance based on' + sampling_factors

    title_elev = 'Elevation Variance post Cloud Filtration based on' + sampling_factors
    title_zen = 'Zenith Angle Variance post Cloud Filtration based on' + sampling_factors

    
    fig, ax = plt.subplots(figsize=(20,16), ncols = 1, nrows = 2, gridspec_kw = None)
    countries = gpd.read_file(  
    gpd.datasets.get_path("naturalearth_lowres"))

    for i in range(2):
        countries.plot(color="lightblue", ax = ax[i])
        
    # plot points
    df.plot.scatter(x="lat", y="lon", s = 8, c=df['elev med'], colormap="plasma", title= title_elev, ax=ax[0])
    df.plot.scatter(x="lat", y="lon", s = 8, c=df['zen'], colormap="plasma", title= title_zen, ax=ax[1])

def wv_range_med_variance_plt(df, sampling_factors):
    title_elev = 'Global Map of Water Vapor Range Variance based on' + sampling_factors
    title_zen = 'Global Map of Water Vapor Median Variance based on' + sampling_factors
    fig, ax = plt.subplots(figsize=(20,16), ncols = 1, nrows = 2, gridspec_kw = None)
    countries = gpd.read_file(  
    gpd.datasets.get_path("naturalearth_lowres"))

    for i in range(2):
        countries.plot(color="lightblue", ax = ax[i])
        
    # plot points
    df.plot.scatter(x="lat", y="lon", s = 8, c=df['wv range'], colormap="plasma", title= title_elev, ax=ax[0])
    df.plot.scatter(x="lat", y="lon", s = 8, c=df['wv med'], colormap="plasma", title= title_zen, ax=ax[1])



    
