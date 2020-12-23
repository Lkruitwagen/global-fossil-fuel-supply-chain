import logging, os, sys, pickle, json, time, yaml
from datetime import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import subprocess

import networkx as nx
import pandas as pd
from math import pi
import numpy as np
from kedro.io import DataCatalog

from shapely import geometry, wkt, ops
from ffsc.pipeline.nodes.utils import V_inv
from ffsc.pipeline.nodes.mp_sjoin_mindist import *

N_WORKERS=6

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def missinglink_cities(df_city_edges, df_cities, df_ports):
    logger=logging.getLogger('Missing cities')
    logger.info('Loading match areas')
    # match cities to ports within match area
    match_areas = pd.DataFrame(json.load(open(os.path.join(os.getcwd(),'results','intersection_geoms.json'),'r')), columns=['geometry'])
    match_areas['geometry'] = match_areas['geometry'].apply(wkt.loads)
    match_areas = gpd.GeoDataFrame(match_areas, geometry='geometry')
    
    # prep ports
    logger.info('prepping Ports')
    df_ports['geometry'] = df_ports['geometry'].apply(wkt.loads)
    df_ports = gpd.GeoDataFrame(df_ports, geometry='geometry').drop(columns=['features','buffer_geom'])
    df_ports['port_geometry'] = df_ports['geometry']
    
    # prep cities
    logger.info('Prepping cities')
    df_cities['geometry'] = df_cities['geometry'].apply(wkt.loads)
    df_cities['geometry'] = df_cities['geometry'].apply(lambda el: el.representative_point())
    df_cities['city_geometry'] = df_cities['geometry']
    df_cities = df_cities.drop(columns=['buffer_geom','features'])
    df_cities = gpd.GeoDataFrame(df_cities, geometry='geometry')
    
    # spatial join
    match_areas = gpd.sjoin(match_areas.reset_index(), df_cities) # first sjoin
    match_areas = gpd.sjoin(match_areas.drop(columns=['index_right']), df_ports) # second sjoin
    
    # map distance
    logger.info('matching distance')
    match_areas['DISTANCE'] = match_areas.apply(lambda row: V_inv((row['city_geometry'].y,row['city_geometry'].x),(row['port_geometry'].y, row['port_geometry'].x))[0]*1000, axis=1) 
    
    print (df_city_edges.append(match_areas.rename(columns={'unique_id_left':'START','unique_id_right':'END'})[['START','END','DISTANCE']]))
    
    return df_city_edges.append(match_areas.rename(columns={'unique_id_left':'START','unique_id_right':'END'})[['START','END','DISTANCE']])


def missinglink_powerstations(df_missing_powerstations_coal, df_missing_powerstations_oil, df_missing_powerstations_gas, df_powerstation_edges, df_city_data):
    logger = logging.getLogger('Missing powerstations')
    
    print ('df_powerstation_edges')
    print (df_powerstation_edges)
    # match missing powerstations to nearest city
    df_missing_powerstations = pd.concat([df_missing_powerstations_coal, df_missing_powerstations_oil, df_missing_powerstations_gas])
    print ('missing_powerstations')
    print (df_missing_powerstations)
    
    logger.info('managing geometries')
    df_missing_powerstations['geometry'] = df_missing_powerstations['geometry'].apply(wkt.loads)
    df_missing_powerstations['buffer_geometry'] = df_missing_powerstations['geometry'].apply(lambda el: el.buffer(5).wkt)
    df_missing_powerstations['geometry'] = df_missing_powerstations['geometry'].apply(lambda el: el.wkt)
    
    logger.info('mp sjoin')
    intersection_df = pd.DataFrame(
                            mp_sjoin_mindist(
                                df_linear=df_city_data, 
                                df_buffered=df_missing_powerstations, 
                                left_geom_column='geometry', 
                                right_geom_column='geometry', 
                                right_buffer_column='buffer_geometry',
                                N_workers=N_WORKERS, 
                                logger=logger,
                                include_min_dist=True),
                            columns=['L_idx','R_idx','intersects', 'NEAREST_PT', 'DISTANCE'])  

    logger.info('sjoin post')
    intersection_df = intersection_df[intersection_df['intersects']==True]
    
    intersection_df = intersection_df.sort_values("DISTANCE").groupby("L_idx", as_index=False).first()
    
    intersection_df['START'] = df_city_data.iloc[intersection_df['R_idx'].values,df_city_data.columns.get_loc('unique_id')].values
    intersection_df['END'] = df_missing_powerstations.iloc[intersection_df['L_idx'].values,df_missing_powerstations.columns.get_loc('unique_id')].values
    
    print ('new df')
    print (df_powerstation_edges.append(intersection_df[['START','END','NEAREST_PT','DISTANCE']]))
    
    return df_powerstation_edges.append(intersection_df[['START','END','NEAREST_PT','DISTANCE']])