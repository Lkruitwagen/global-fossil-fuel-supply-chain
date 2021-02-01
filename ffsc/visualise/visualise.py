import logging, os, sys, pickle, json, time, yaml
from datetime import datetime as dt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
tqdm.pandas()

import pandas as pd
import geopandas as gpd
from geopandas.plotting import _plot_linestring_collection, _plot_point_collection
import numpy as np

from shapely import geometry, wkt, ops
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from ffsc.pipeline.nodes.utils import V_inv

import networkx as nx

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import multiprocessing as mp
N_WORKERS=6
    
def visualise_gpd(params, gdfs, ne, logger):
    fig, ax = plt.subplots(1,1,figsize=params['figsize'])
    ne.plot(ax=ax, color='#{:02x}{:02x}{:02x}'.format(*params['vis_colors']['ne']), **params['type_style']['ne'])
    
    for dd in gdfs:
        logger.info(f'plotting {dd["type"]} {dd["color_key"]}')
        if dd['type']=='lin_asset':
            dd['gdf']['len'] = dd['gdf']['geometry'].apply(lambda geom: geom.length)
            dd['gdf'] = dd['gdf'][dd['gdf']['len']<345]
        dd['gdf'].plot(
            ax=ax, 
            color='#{:02x}{:02x}{:02x}'.format(*params['vis_colors'][dd['color_key']]),
            **params['type_style'][dd['type']]
        )
    
    plt.savefig(params['path'])
    
    
def visualise_assets_simplified_coal(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne):
    params['path'] = os.path.join(os.getcwd(),'results','figures','assets_simplified_coal.png')
    visualise_assets(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne)
    return []
    
def visualise_assets_simplified_oil(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne):
    params['path'] = os.path.join(os.getcwd(),'results','figures','assets_simplified_oil.png')
    visualise_assets(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne)
    return []
    
def visualise_assets_simplified_gas(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne):
    params['path'] = os.path.join(os.getcwd(),'results','figures','assets_simplified_gas.png')
    visualise_assets(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne)
    return []
    
def visualise_assets_coal(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne):
    params['path'] = os.path.join(os.getcwd(),'results','figures','assets_coal.png')
    visualise_assets(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne)
    return []
    
def visualise_assets_oil(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne):
    params['path'] = os.path.join(os.getcwd(),'results','figures','assets_oil.png')
    visualise_assets(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne)
    return []
    
def visualise_assets_gas(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne):
    params['path'] = os.path.join(os.getcwd(),'results','figures','assets_gas.png')
    visualise_assets(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne)
    return []

def prep_assets(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne):
    logger = logging.getLogger('Prep assets')
    df_ptassets = pd.concat([refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations])
    df_linassets = pd.concat([railways, shippingroutes,pipelines])
    
    ### filter all dfs
    all_nodes = list(set(df_edges['source'].unique().tolist() + df_edges['target'].unique().tolist()))
    #all_nodes = all_nodes + [n+'_B' for n in all_nodes]
    
    # drop the lin assets 
    logger.info('Dropping nodes not in edges')
    df_edges['source_type'] = df_edges['source'].str.split('_').str[0]
    df_edges['target_type'] = df_edges['target'].str.split('_').str[0]
    for lin_asset in ['RAILWAY','PIPELINE','SHIPPINGROUTE']:
        df_edges = df_edges[~((df_edges['source_type']==lin_asset) & (df_edges['target_type']==lin_asset))]
    df_edges = df_edges[df_edges['source_type']!='supersource']
    
    # drop any '_B' assets
    df_edges['source'] = df_edges['source'].str.replace('_B','')
    df_edges['target'] = df_edges['target'].str.replace('_B','')
    
    # join geometries for missing
    df_missing_cities = pd.merge(df_missing_cities, df_ptassets[['unique_id','geometry']], how='left',on='unique_id')
    
    #print ('missing')
    #print (df_missing_cities)
    #print (df_missing_powerstations)
    
    # drop non-nodes
    df_ptassets = df_ptassets[df_ptassets['unique_id'].isin(all_nodes)]
    df_linassets = df_linassets[df_linassets['START'].isin(all_nodes)]
    
    # map geoms on ptassets
    logger.info('mapping geometries')
    df_ptassets['geometry'] = df_ptassets['geometry'].apply(wkt.loads)
    # do polygon assets
    df_ptassets.loc[df_ptassets['unique_id'].str.split('_').str[0]=='OILFIELD','geometry'] = df_ptassets.loc[df_ptassets['unique_id'].str.split('_').str[0]=='OILFIELD','geometry'].apply(lambda el: el.representative_point())
    df_ptassets.loc[df_ptassets['unique_id'].str.split('_').str[0]=='CITY','geometry'] = df_ptassets.loc[df_ptassets['unique_id'].str.split('_').str[0]=='CITY','geometry'].apply(lambda el: el.representative_point())
    
    # map geoms on remaining edges
    df_edges = pd.merge(df_edges, df_ptassets[['unique_id','geometry']], how='left',left_on='source',right_on='unique_id').rename(columns={'geometry':'geometry_source'}).drop(columns=['unique_id'])
    df_edges = pd.merge(df_edges, df_ptassets[['unique_id','geometry']], how='left',left_on='target',right_on='unique_id').rename(columns={'geometry':'geometry_target'}).drop(columns=['unique_id'])
    
    df_edges.loc[df_edges['source_type'].isin(['RAILWAY','PIPELINE','SHIPPINGROUTE']),'geometry_source'] = df_edges.loc[df_edges['source_type'].isin(['RAILWAY','PIPELINE','SHIPPINGROUTE']),'source'].apply(lambda el: geometry.Point([float(cc) for cc in el.split('_')[2:4]]))
    df_edges.loc[df_edges['target_type'].isin(['RAILWAY','PIPELINE','SHIPPINGROUTE']),'geometry_target'] = df_edges.loc[df_edges['target_type'].isin(['RAILWAY','PIPELINE','SHIPPINGROUTE']),'target'].apply(lambda el: geometry.Point([float(cc) for cc in el.split('_')[2:4]]))
    
    print ('bork')
    print (df_edges.loc[df_edges['target_type'].isin(['RAILWAY','PIPELINE','SHIPPINGROUTE'])])
    
    df_edges['geometry'] = df_edges.apply(lambda row: geometry.LineString([row['geometry_source'],row['geometry_target']]), axis=1)
    
    print ('IDL')
    pos_idl = ((df_linassets['START'].str.split('_').str[0]=='SHIPPINGROUTE') &(df_linassets['END'].str.split('_').str[0]=='SHIPPINGROUTE')&(df_linassets['START'].str.split('_').str[2].astype(float)<-175)&(df_linassets['END'].str.split('_').str[2].astype(float)>175))
    neg_idl =((df_linassets['START'].str.split('_').str[0]=='SHIPPINGROUTE') &(df_linassets['END'].str.split('_').str[0]=='SHIPPINGROUTE')&(df_linassets['START'].str.split('_').str[2].astype(float)>175)&(df_linassets['END'].str.split('_').str[2].astype(float)<-175))
    print (pos_idl.sum(), neg_idl.sum())
    
    # remove IDL from linassets
    df_linassets = df_linassets[~pos_idl]
    df_linassets = df_linassets[~neg_idl]
    
    # map geoms on linassets (LSS)
    df_linassets['start_geometry'] = df_linassets['START'].apply(lambda el: geometry.Point([float(cc) for cc in el.split('_')[2:4]]))
    df_linassets['end_geometry'] = df_linassets['END'].apply(lambda el: geometry.Point([float(cc) for cc in el.split('_')[2:4]]))
    df_linassets['geometry'] = df_linassets.apply(lambda row: geometry.LineString([row['start_geometry'],row['end_geometry']]),axis=1)
    
    # map geoms on missing
    df_missing_cities['geometry'] = df_missing_cities['geometry'].apply(wkt.loads)
    df_missing_cities['geometry'] = df_missing_cities['geometry'].apply(lambda el: el.representative_point())
    df_missing_powerstations['geometry'] = df_missing_powerstations['geometry'].apply(wkt.loads)
    
    print ('edges')
    print (df_edges)
    
    print ('assets')
    print (df_ptassets)
    
    print ('linassets')
    print (df_linassets)
    
    print ('tuples')
    print (set([tuple(el) for el in df_edges[['source_type','target_type']].values.tolist()]))
    
    # get color keys
    df_edges['color_key'] = 'FINALMILE'
    for kk in ['RAILWAY','PIPELINE','SHIPPINGROUTE']:
        df_edges.loc[((df_edges['source_type']==kk) | (df_edges['target_type']==kk)),'color_key'] = kk
    df_linassets['color_key'] = df_linassets['START'].str.split('_').str[0]
    df_ptassets['color_key'] = df_ptassets['unique_id'].str.split('_').str[0]
    
    return df_edges, df_linassets, df_ptassets, df_missing_cities, df_missing_powerstations

def visualise_assets(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways,shippingroutes, pipelines, ne):
    logger=logging.getLogger('Visualising')
    
    df_edges, df_linassets, df_ptassets, df_missing_cities, df_missing_powerstations = prep_assets(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_missing_cities, df_missing_powerstations, railways, shippingroutes, pipelines, ne)
    
    
    # prep gdfs
    logger.info('Prepping geodataframes')
    gdfs = []
    
    for kk in df_ptassets['color_key'].unique():
        gdfs.append(
            {
                'gdf':gpd.GeoDataFrame(df_ptassets[df_ptassets['color_key']==kk], geometry='geometry'),
                'color_key':kk,
                'type':'pt_asset'
            }
        )
        
    for kk in df_linassets['color_key'].unique():
        gdfs.append(
            {
                'gdf':gpd.GeoDataFrame(df_linassets[df_linassets['color_key']==kk], geometry='geometry'),
                'color_key':kk,
                'type':'lin_asset'
            }
        )
        
    for kk in df_edges['color_key'].unique():
        gdfs.append(
            {
                'gdf':gpd.GeoDataFrame(df_edges[df_edges['color_key']==kk], geometry='geometry'),
                'color_key':kk,
                'type':'edges'
            }
        )
    # missing
    gdfs += [
        {
            'gdf':gpd.GeoDataFrame(df_missing_cities, geometry='geometry'),
            'color_key':'MISSING_CITY',
            'type':'missing_city',
        },
        {
            'gdf':gpd.GeoDataFrame(df_missing_powerstations, geometry='geometry'),
            'color_key':'MISSING_POWERSTATION',
            'type':'missing_powerstation',
        },
    ]
    params['figsize'] = (72,48)
    
    logger.info('Callign mpl')
    visualise_gpd(params, gdfs, ne, logger)
    

    return []

def visualise_flow(params, ne, df_flow, df_community_edges, df_community_nodes):
    
    # get carrier
    if 'COALMINE' in df_community_nodes['NODETYPE'].unique():
        carrier='coal'
        carrier_supplytypes = ['COALMINE']
    elif 'LNGTERMINAL' in df_community_nodes['NODETYPE'].unique():
        carrier='gas'
        carrier_supplytypes = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        carrier_supplytypes = ['OILFIELD','OILWELL']
        
    logger = logging.getLogger(f'visualise flow: {carrier}')
    
    logger.info('prepping DFs')
    df_community_nodes = df_community_nodes[~df_community_nodes['NODETYPE'].isin(['RAILWAY','PIPELINE','SHIPPINGROUTE'])]
    print ('nodes')
    print (df_community_nodes)
    df_flow = df_flow.rename(columns={'SOURCE':'source','TARGET':'target'})
    df_flow = df_flow.set_index(['source','target'])
    print ('df_flow')
    print (df_flow)
    df_community_edges['source_type'] = df_community_edges['source'].str.split('_').str[0]
    df_community_edges['target_type'] = df_community_edges['target'].str.split('_').str[0]
    df_community_edges = df_community_edges.set_index(['source','target'])
    print ('df edges')
    print (df_community_edges)
    df_community_edges = pd.merge(df_community_edges, df_flow[['flow']], how='left', left_index=True, right_index=True)
    
    logger.info('mapping geometries')
    df_community_edges['geometry'] = df_community_edges['geometry'].apply(wkt.loads)
    df_community_nodes['geometry'] = df_community_nodes['geometry'].apply(wkt.loads)
    
    logger.info('doing colors and weights')
    #df_colors = pd.DataFrame.from_dict({kk:"#{:02x}{:02x}{:02x}".format(*vv) for kk,vv in params['vis_colors'].items()}, orient='index').rename(columns={0:'hex'})
    colormap = {kk:"#{:02x}{:02x}{:02x}".format(*vv) for kk,vv in params['vis_colors'].items()}
    
    df_community_edges['color_key'] = 'FINALMILE'
    for kk in ['RAILWAY','PIPELINE','SHIPPINGROUTE']:
        df_community_edges.loc[((df_community_edges['source_type']==kk) | (df_community_edges['target_type']==kk)),'color_key'] = kk
        
        
    df_community_edges['color_hex'] = df_community_edges['color_key'].map(colormap)
    df_community_nodes['color_hex'] = df_community_nodes['NODETYPE'].map(colormap)
    
    MIN_EDGE = 1
    MAX_EDGE = 10
    MIN_NODE = 1
    MAX_NODE = 25
    
    df_community_nodes = pd.merge(df_community_nodes, df_flow.reset_index()[['target','flow']], how='left',left_on='NODE',right_on='target')
    # do demand and supply separately
    df_community_nodes['s'] = (np.log10(df_community_nodes['D']+1) - np.log10(df_community_nodes['D']+1).min())/(np.log10(df_community_nodes['D']+1).max() - np.log10(df_community_nodes['D']+1).min())*(MAX_NODE-MIN_NODE)+MIN_NODE
    df_community_nodes['s_flow'] = (np.log10(df_community_nodes['flow']+1) - np.log10(df_community_nodes['D']+1).min())/(np.log10(df_community_nodes['flow']+1).max() - np.log10(df_community_nodes['flow']+1).min())*(MAX_NODE-MIN_NODE)+MIN_NODE
    df_community_nodes.loc[df_community_nodes['NODETYPE'].isin(carrier_supplytypes),'s'] = df_community_nodes.loc[df_community_nodes['NODETYPE'].isin(carrier_supplytypes),'s_flow']
    
    #df_community_edges['s'] = (np.log(df_community_edges['flow']+1) - np.log(df_community_edges['flow']+1).min())/(np.log(df_community_edges['flow']+1).max() - np.log(df_community_edges['flow']+1).min())*(MAX_EDGE-MIN_EDGE)+MIN_EDGE
    df_community_edges['s'] = (df_community_edges['flow'] - df_community_edges['flow'].min())/(df_community_edges['flow'].max() - df_community_edges['flow'].min())*(MAX_EDGE-MIN_EDGE)+MIN_EDGE
    
    df_community_edges = df_community_edges[df_community_edges['flow']>0]
    
    
    # get rid of the ones that are super long
    df_community_edges['len'] = df_community_edges['geometry'].apply(lambda geom: geom.length)
    df_community_edges = df_community_edges[df_community_edges['len']<350]
    
    #cast to gdf
    df_community_nodes = gpd.GeoDataFrame(df_community_nodes, geometry='geometry')
    df_community_edges = gpd.GeoDataFrame(df_community_edges, geometry='geometry')
    
    fig, ax = plt.subplots(1,1,figsize=(72,48))
    ne.plot(ax=ax, color='#{:02x}{:02x}{:02x}'.format(*params['vis_colors']['ne']), **params['type_style']['ne'])

    
    _plot_point_collection(
        ax=ax,
        geoms=df_community_nodes['geometry'],
        color=df_community_nodes['color_hex'].values.tolist(),
        markersize=df_community_nodes['s'].values.tolist()
    )
    _plot_linestring_collection(
        ax=ax,
        geoms=df_community_edges['geometry'],
        color=df_community_edges['color_hex'].values.tolist(),
        linewidth=df_community_edges['s'].values.tolist()
    )
    
    plt.savefig(os.path.join(os.getcwd(),'results','figures',f'flow_{carrier}.png'))
    
    return []


def compare_flow(params, ne, df_flow_bl, df_flow_cf, df_community_edges, df_community_nodes):
    
    # get carrier
    if 'COALMINE' in df_community_nodes['NODETYPE'].unique():
        carrier='coal'
        carrier_supplytypes = ['COALMINE']
    elif 'LNGTERMINAL' in df_community_nodes['NODETYPE'].unique():
        carrier='gas'
        carrier_supplytypes = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        carrier_supplytypes = ['OILFIELD','OILWELL']
        
    logger = logging.getLogger(f'visualise flow: {carrier}')
    writer = logging.getLogger(f'writer_{carrier}')
    fh = logging.FileHandler(f'compare_{carrier}.log')
    fh.setLevel(logging.INFO)
    writer.addHandler(fh)
    
    logger.info('prepping DFs')
    df_community_nodes = df_community_nodes[~df_community_nodes['NODETYPE'].isin(['RAILWAY','PIPELINE','SHIPPINGROUTE'])]
    df_flow_bl = df_flow_bl.rename(columns={'SOURCE':'source','TARGET':'target'})
    df_flow_bl = df_flow_bl.set_index(['source','target'])
    df_flow_cf = df_flow_cf.rename(columns={'SOURCE':'source','TARGET':'target'})
    df_flow_cf = df_flow_cf.set_index(['source','target'])
    df_community_edges['source_type'] = df_community_edges['source'].str.split('_').str[0]
    df_community_edges['target_type'] = df_community_edges['target'].str.split('_').str[0]
    df_community_edges = df_community_edges.set_index(['source','target'])
    print ('edges')
    print (df_community_edges)
    print ('flow_bl')
    print(df_flow_bl)
    print ('flow_cf')
    print(df_flow_cf)
    df_community_edges = pd.merge(df_community_edges, df_flow_bl[['flow']], how='left', left_index=True, right_index=True).rename(columns={'flow':'bl_flow'})
    df_community_edges = pd.merge(df_community_edges, df_flow_cf[['flow']], how='left', left_index=True, right_index=True).rename(columns={'flow':'cf_flow'})
    
    logger.info('mapping geometries')
    df_community_edges['geometry'] = df_community_edges['geometry'].apply(wkt.loads)
    df_community_nodes['geometry'] = df_community_nodes['geometry'].apply(wkt.loads)
    
    logger.info('doing colors and weights')
    #df_colors = pd.DataFrame.from_dict({kk:"#{:02x}{:02x}{:02x}".format(*vv) for kk,vv in params['vis_colors'].items()}, orient='index').rename(columns={0:'hex'})
    colormap = {kk:"#{:02x}{:02x}{:02x}".format(*vv) for kk,vv in params['vis_colors'].items()}
    
    #df_community_edges['color_key'] = 'FINALMILE'
    #for kk in ['RAILWAY','PIPELINE','SHIPPINGROUTE']:
    #    df_community_edges.loc[((df_community_edges['source_type']==kk) | (df_community_edges['target_type']==kk)),'color_key'] = kk
        
        
    #df_community_edges['color_hex'] = df_community_edges['color_key'].map(colormap)
    df_community_nodes['color_hex'] = df_community_nodes['NODETYPE'].map(colormap)
    
    MIN_EDGE = 1
    MAX_EDGE = 10
    MIN_NODE = 1
    MAX_NODE = 25
    
    df_community_nodes = pd.merge(df_community_nodes, df_flow_bl.reset_index()[['target','flow']], how='left',left_on='NODE',right_on='target')
    # do demand and supply separately
    df_community_nodes['s'] = (np.log10(df_community_nodes['D']+1) - np.log10(df_community_nodes['D']+1).min())/(np.log10(df_community_nodes['D']+1).max() - np.log10(df_community_nodes['D']+1).min())*(MAX_NODE-MIN_NODE)+MIN_NODE
    df_community_nodes['s_flow'] = (np.log10(df_community_nodes['flow']+1) - np.log10(df_community_nodes['D']+1).min())/(np.log10(df_community_nodes['flow']+1).max() - np.log10(df_community_nodes['flow']+1).min())*(MAX_NODE-MIN_NODE)+MIN_NODE
    df_community_nodes.loc[df_community_nodes['NODETYPE'].isin(carrier_supplytypes),'s'] = df_community_nodes.loc[df_community_nodes['NODETYPE'].isin(carrier_supplytypes),'s_flow']
    
    #df_community_edges['s'] = (np.log(df_community_edges['flow']+1) - np.log(df_community_edges['flow']+1).min())/(np.log(df_community_edges['flow']+1).max() - np.log(df_community_edges['flow']+1).min())*(MAX_EDGE-MIN_EDGE)+MIN_EDGE
    df_community_edges['s'] = (df_community_edges['bl_flow'] - df_community_edges['bl_flow'].min())/(df_community_edges['bl_flow'].max() - df_community_edges['bl_flow'].min())*(MAX_EDGE-MIN_EDGE)+MIN_EDGE
    
    print('new edges')
    print (df_community_edges.loc[(df_community_edges['cf_flow']>0) & (df_community_edges['bl_flow']==0)])
    df_community_edges['difference'] = df_community_edges['bl_flow'] - df_community_edges['cf_flow']
    
    df_community_edges['reduction'] = df_community_edges['difference']/df_community_edges['bl_flow']
    
    cm = LinearSegmentedColormap.from_list('GrayRd', [(0,1,0),(.63,.63,.63),(1, 0, 0)], N=255)
    
    df_community_edges = df_community_edges[(df_community_edges['bl_flow']>0) | (df_community_edges['cf_flow'])>0]
    
    def apply_colmap(row):
        if row['bl_flow']>0:
            cm_val = (row['reduction']+1.)*128 # between 0 and 255 with 128 as neutral
            return '#{:02x}{:02x}{:02x}'.format(*[int(il*255) for il in cm(int(cm_val))[0:3]])
        else:
            return '#{:02x}{:02x}{:02x}'.format(0,0,255)
    
    df_community_edges['color_hex'] = df_community_edges.apply(lambda row: apply_colmap(row), axis=1) 
    
    
    # filter IDL -> just use euclidean length
    logger.info('remove idl edges')
    df_community_edges['len'] = df_community_edges['geometry'].apply(lambda el: el.length)
    df_community_edges = df_community_edges[df_community_edges['len']<=350]
    
    df_community_edges = df_community_edges.reset_index()
    
    # get top changes and add write them to file
    # want: top/bottom 10 reduced sources
    logger.info('writing differences to file')
    for idx, val in df_community_edges.loc[df_community_edges['source_type'].isin(carrier_supplytypes),['source','difference','bl_flow']].groupby('source').sum().sort_values('difference').iloc[:10].iterrows():
        writer.info(f'idx:{idx}\t difference:{val["difference"]}\t bl_flow:{val["bl_flow"]}')
    for idx, val in df_community_edges.loc[df_community_edges['source_type'].isin(carrier_supplytypes),['source','difference','bl_flow']].groupby('source').sum().sort_values('difference').iloc[-10:].iterrows():
        writer.info(f'idx:{idx}\t difference:{val["difference"]}\t bl_flow:{val["bl_flow"]}')
        
    # want: top/bottom 10 reduced transmission
    for idx, val in df_community_edges.loc[~df_community_edges['source_type'].isin(carrier_supplytypes),['difference', 'reduction']].sort_values('difference').iloc[:10].iterrows():
        writer.info(f'idx:{idx}\t difference:{val["difference"]}\t reduction: {val["reduction"]}')
    for idx, val in df_community_edges.loc[~df_community_edges['source_type'].isin(carrier_supplytypes),['difference','reduction']].sort_values('difference').iloc[-10:].iterrows():
        writer.info(f'idx:{idx}\t difference:{val["difference"]}\t reduction: {val["reduction"]}')
    
    
    
    #cast to gdf
    df_community_nodes = gpd.GeoDataFrame(df_community_nodes, geometry='geometry')
    df_community_edges = gpd.GeoDataFrame(df_community_edges, geometry='geometry')
    
    fig, ax = plt.subplots(1,1,figsize=(72,48))
    ne.plot(ax=ax, color='#{:02x}{:02x}{:02x}'.format(*params['vis_colors']['ne']), **params['type_style']['ne'])

    
    _plot_point_collection(
        ax=ax,
        geoms=df_community_nodes['geometry'],
        color=df_community_nodes['color_hex'].values.tolist(),
        markersize=df_community_nodes['s'].values.tolist()
    )
    _plot_linestring_collection(
        ax=ax,
        geoms=df_community_edges['geometry'],
        color=df_community_edges['color_hex'].values.tolist(),
        linewidth=df_community_edges['s'].values.tolist()
    )
    
    plt.savefig(os.path.join(os.getcwd(),'results','figures',f'flow_sds_{carrier}.png'))
    
    return []


def node_iso2(iso2,ne,df_coal_nodes,df_oil_nodes,df_gas_nodes,df_raw_oilfields,df_raw_oilwells):
    logger=logging.getLogger('do iso2s')
    
    logger.info('Doing oilfields and oilwells which might be  offshore')
    
    map_dict = iso2[['country','iso2']].set_index('country').to_dict()['iso2']
    oilwells_map = {'congo':'CG',"cote d''ivoire":'CI','iran (islamic republic of)':'IR'}
    oilfields_map = {
           'venezuela, bolivarian republic of':'VE',
           'syrian arab republic':'SY',
           'democratic republic of the congo':'CD',
           'republic of korea':'KR',
           "democratic people''s republic of korea":'KP',
           'united republic of tanzania':'TZ', 
           'republic of korea':'KR',
           'guinea bissau':'GW',
           'bolivia (plurinational state of)':'BO',
           'republic of moldova':'MD',
           'the former yugoslav republic of macedonia':'MK', 
           "lao people''s democratic republic":'LA',
    }
    map_dict.update(oilwells_map)
    map_dict.update(oilfields_map)
    
    df_raw_oilfields['md_country'] = df_raw_oilfields['md_country'].str.lower()
    df_raw_oilwells['md_country'] = df_raw_oilwells['md_country'].str.lower()
    iso2['country'] = iso2['country'].str.lower()
    
    df_raw_oilfields = pd.merge(df_raw_oilfields, iso2[['country','iso2']], how='left',left_on='md_country',right_on='country')
    #df_raw_oilfields.loc[df_raw_oilfields['iso2'].isna(),'iso2'] = df_raw_oilfields.loc[df_raw_oilfields['iso2'].isna(),'md_country'].str.split(';').apply(lambda ll: ','.join([map_dict[el.strip()] if el.strip() in map_dict.keys() else el for el in ll]))
    
    df_raw_oilwells = pd.merge(df_raw_oilwells, iso2[['country','iso2']], how='left',left_on='md_country',right_on='country')
    df_raw_oilwells.loc[df_raw_oilwells['iso2'].isna(),'iso2'] = df_raw_oilwells.loc[df_raw_oilwells['iso2'].isna(),'md_country'].map(map_dict)
    
    df_raw_oilwells = df_raw_oilwells.reset_index()
    df_raw_oilfields = df_raw_oilfields.reset_index()
    df_raw_oilwells['unique_id'] = 'OILWELL_'+df_raw_oilwells['index'].astype(str)
    df_raw_oilfields['unique_id'] = 'OILFIELD_'+df_raw_oilfields['index'].astype(str)
    
    # drop rail, shipping, pipeline
    
    logger.info('doing geometries')
    df_coal_nodes = df_coal_nodes[~df_coal_nodes['NODETYPE'].isin(['RAILWAY','SHIPPINGROUTE','PIPELINE'])]
    df_oil_nodes = df_oil_nodes[~df_oil_nodes['NODETYPE'].isin(['RAILWAY','SHIPPINGROUTE','PIPELINE'])]
    df_gas_nodes = df_gas_nodes[~df_gas_nodes['NODETYPE'].isin(['RAILWAY','SHIPPINGROUTE','PIPELINE'])]
    
    # load geometry
    df_coal_nodes['geometry'] = df_coal_nodes['geometry'].apply(wkt.loads)
    df_oil_nodes['geometry'] = df_oil_nodes['geometry'].apply(wkt.loads)
    df_gas_nodes['geometry'] = df_gas_nodes['geometry'].apply(wkt.loads)
    
    logger.info('Doing sjoin')
    df_coal_nodes = pd.DataFrame(gpd.sjoin(gpd.GeoDataFrame(df_coal_nodes,geometry='geometry'),ne[['ISO_A2','geometry']],how='left'))#.rename(columns={'ISO_A2':'iso2'})
    df_oil_nodes = pd.DataFrame(gpd.sjoin(gpd.GeoDataFrame(df_oil_nodes,geometry='geometry'),ne[['ISO_A2','geometry']],how='left'))#.rename(columns={'ISO_A2':'iso2'})
    df_gas_nodes = pd.DataFrame(gpd.sjoin(gpd.GeoDataFrame(df_gas_nodes,geometry='geometry'),ne[['ISO_A2','geometry']],how='left'))#.rename(columns={'ISO_A2':'iso2'})
    
    ### now merge on oilwells and oilfields
    df_coal_nodes = pd.merge(df_coal_nodes, df_raw_oilwells[['unique_id','iso2']], how='left',left_on='NODE',right_on='unique_id').rename(columns={'iso2':'iso2_ow'})
    df_coal_nodes = pd.merge(df_coal_nodes, df_raw_oilfields[['unique_id','iso2']], how='left',left_on='NODE',right_on='unique_id').rename(columns={'iso2':'iso2_of'})
    df_oil_nodes = pd.merge(df_oil_nodes, df_raw_oilwells[['unique_id','iso2']], how='left',left_on='NODE',right_on='unique_id').rename(columns={'iso2':'iso2_ow'})
    df_oil_nodes = pd.merge(df_oil_nodes, df_raw_oilfields[['unique_id','iso2']], how='left',left_on='NODE',right_on='unique_id').rename(columns={'iso2':'iso2_of'})
    df_gas_nodes = pd.merge(df_gas_nodes, df_raw_oilwells[['unique_id','iso2']], how='left',left_on='NODE',right_on='unique_id').rename(columns={'iso2':'iso2_ow'})
    df_gas_nodes = pd.merge(df_gas_nodes, df_raw_oilfields[['unique_id','iso2']], how='left',left_on='NODE',right_on='unique_id').rename(columns={'iso2':'iso2_of'})
    
    df_coal_nodes['ISO_A2'] = df_coal_nodes['ISO_A2'].fillna(df_coal_nodes['iso2_ow']).fillna(df_coal_nodes['iso2_of'])
    df_oil_nodes['ISO_A2'] = df_oil_nodes['ISO_A2'].fillna(df_oil_nodes['iso2_ow']).fillna(df_oil_nodes['iso2_of'])
    df_gas_nodes['ISO_A2'] = df_gas_nodes['ISO_A2'].fillna(df_gas_nodes['iso2_ow']).fillna(df_gas_nodes['iso2_of'])
    
    df_coal_nodes = df_coal_nodes.drop(columns=['iso2_of','iso2_ow']).rename(columns={'ISO_A2':'iso2'})
    df_oil_nodes = df_oil_nodes.drop(columns=['iso2_of','iso2_ow']).rename(columns={'ISO_A2':'iso2'})
    df_gas_nodes = df_gas_nodes.drop(columns=['iso2_of','iso2_ow']).rename(columns={'ISO_A2':'iso2'})
    
    ### finally fillna with missing
    
    def mindist(row):
        
        def _mindist(pt1,pt2):
            try:
                return V_inv((pt1.y,pt1.x),(pt2.y, pt2.x))[0]*1000
            except:
                return np.inf
        
        ne['NEAREST_PT'] = ne['geometry'].apply(lambda geom: ops.nearest_points(geom,row['geometry'])[0])
        ne['DIST'] = ne['NEAREST_PT'].apply(lambda pt: _mindist(pt,row['geometry'])) #V_inv((50,179),(50,-175))
        min_idx = ne['DIST'].idxmin()
        
        return ne.iloc[min_idx, ne.columns.get_loc('ISO_A2')]
    
    do_nodetypes = ['COALMINE','OILFIELD','OILWELL','CITY','POWERSTATION']
    
    logger.info('Doing last manually')
    df_coal_nodes.loc[df_coal_nodes['iso2'].isna() & df_coal_nodes['NODETYPE'].isin(do_nodetypes),'iso2'] = df_coal_nodes.loc[df_coal_nodes['iso2'].isna()  & df_coal_nodes['NODETYPE'].isin(do_nodetypes),:].progress_apply(lambda row: mindist(row), axis=1)
    df_oil_nodes.loc[df_oil_nodes['iso2'].isna()  & df_oil_nodes['NODETYPE'].isin(do_nodetypes),'iso2'] = df_oil_nodes.loc[df_oil_nodes['iso2'].isna()  & df_oil_nodes['NODETYPE'].isin(do_nodetypes),:].progress_apply(lambda row: mindist(row), axis=1)
    df_gas_nodes.loc[df_gas_nodes['iso2'].isna()  & df_gas_nodes['NODETYPE'].isin(do_nodetypes),'iso2'] = df_gas_nodes.loc[df_gas_nodes['iso2'].isna()  & df_gas_nodes['NODETYPE'].isin(do_nodetypes),:].progress_apply(lambda row: mindist(row), axis=1)
    
    
    print ('missing 3')
    print (df_coal_nodes.loc[df_coal_nodes['iso2'].isna(),'NODETYPE'].unique())
    print (df_oil_nodes.loc[df_oil_nodes['iso2'].isna(),'NODETYPE'].unique())
    print (df_gas_nodes.loc[df_gas_nodes['iso2'].isna(),'NODETYPE'].unique())
    
    
    return df_coal_nodes[['NODE','iso2']], df_oil_nodes[['NODE','iso2']], df_gas_nodes[['NODE','iso2']]


def mp_dijkstra(G, sources, w_id):
    
    recs = []
    print (w_id, 'len sources',len(sources))
    
    for ii_s, source in enumerate(sources):
        if ii_s % 5 ==0:
            print (f'wid:{w_id}, ii_s:{ii_s}, source: {source}')
        dd = nx.single_source_dijkstra_path_length(G, source, cutoff=None, weight='inv_flow')
        recs.append({'source':source,'weight':{kk:vv for kk,vv in dd.items() if kk.split('_')[0] in ['CITY','POWERSTATION']}})
        
        
    return recs


def flow2iso2adj(iso2, df_flow, df_community_iso2):
    
    
    if 'COALMINE' in df_community_iso2['NODE'].str.split('_').str[0].unique():
        carrier='coal'
        carrier_sources=['COALMINE']
    elif 'LNGTERMINAL' in df_community_iso2['NODE'].str.split('_').str[0].unique():
        carrier = 'gas'
        carrier_sources =['OILFIELD']
    else:
        carrier = 'oil'
        carrier_sources = ['OILWELL']
        
    print ('carrier', carrier)
        
    logger = logging.getLogger(f'Visualise-trade_{carrier}')
    logger.info('prepping graph')
    df_flow['inv_flow'] = 1/df_flow['flow'].astype(float)
    df_flow.loc[df_flow['inv_flow']==np.inf,'inv_flow'] = 2*df_flow['flow'].max()
    df_flow['inv_flow'] = df_flow['inv_flow']/df_flow['inv_flow'].min()
    
    print('df flow')
    print (df_flow)
    
    G = nx.DiGraph()
    G.add_edges_from([(r[0],r[1],{'flow':r[2], 'flow_inv':r[3]}) for r in df_flow[['source','target','flow','inv_flow']].values.tolist()])
    
    MI = pd.MultiIndex.from_product([iso2['iso2'].unique().tolist(),iso2['iso2'].unique().tolist()], names=('source', 'target'))
    all_flows = pd.DataFrame(index=MI)
    
    all_flows['flow'] = 0
    
    sources = df_flow.loc[df_flow['target'].str.split('_').str[0].isin(['COALMINE']) & (df_flow['flow']>0),'target'].values.tolist()
    #sources = df_community_iso2.loc[df_community_iso2['NODE'].str.split('_').str[0].isin(carrier_sources),'NODE'].values.tolist()
    #print (sources)
    logger.info(f'Number of sources {len(sources)}')
    
    
    df_community_iso2 = df_community_iso2.set_index('NODE')
    df_flow = df_flow.set_index(['source','target'])
    
    logger.info('calling mp dijkstra')
    
    
    
    size = len(sources)//N_WORKERS+1
    #size=5
    print ('size',size)
    source_bunches = [sources[ii*size:(ii+1)*size] for ii in range(N_WORKERS)]
    
    args = [(G,source_bunches[ii],ii) for ii in range(N_WORKERS)]    
    
    pool = mp.Pool(N_WORKERS)
    
    recs = pool.starmap(mp_dijkstra, args)
    recs =  [item for sublist in recs for item in sublist]
    
    print ('len recs',len(recs))
    
    for rec in recs:
        #print (rec)
        df = pd.DataFrame.from_dict({'weight':{kk:vv for kk,vv in rec['weight'].items()}})
        df = df[df.index.str.split('_').str[0].isin(['POWERSTATION','CITY'])]
        
        df = pd.merge(df, df_community_iso2, how='left',left_index=True, right_index=True)
        df['weight'] = 1/df['weight']
        df['weight'] = df['weight']/df['weight'].sum()
        summary = df.groupby('iso2').sum().reset_index().rename(columns={'iso2':'target'})
        summary['source'] = df_community_iso2.at[rec['source'],'iso2']
        summary['weight'] = summary['weight']*df_flow.at[('supersource',rec['source']),'flow']
        summary = summary.set_index(['source','target'])
        
        all_flows.loc[summary.index,'flow'] = all_flows.loc[summary.index,'flow'] + summary['weight']
        print ('sum',all_flows['flow'].sum())
    
    """
    for source in tqdm(sources):
        
        print (source)
    
        dd = nx.single_source_dijkstra_path_length(G, source, cutoff=None, weight='inv_flow')
        df = pd.DataFrame.from_dict({'weight':{kk:vv for kk,vv in dd.items() if kk.split('_')[0] in ['CITY','POWERSTATION']}})
        df = df[df.index.str.split('_').str[0].isin(['POWERSTATION','CITY'])]
        
        df = pd.merge(df, df_community_iso2, how='left',left_index=True, right_index=True)
        df['weight'] = 1/df['weight']
        df['weight'] = df['weight']/df['weight'].sum()
        summary = df.groupby('iso2').sum().reset_index().rename(columns={'iso2':'target'})
        summary['source'] = df_community_iso2.at[source,'iso2']
        summary['weight'] = summary['weight']*df_flow.at[('supersource',source),'flow']
        summary = summary.set_index(['source','target'])
        
        all_flows.loc[summary.index,'flow'] = all_flows.loc[summary.index,'flow'] + summary['weight']
    """
    print (all_flows)
        
    return all_flows
    


def visualise_trade_arrows(iso2, ne, params, df_energy, df_trade):
    
    if df_trade.iloc[0,df_trade.columns.get_loc('Commodity Code')]==2701:
        carrier='coal'
        carrier_sources=['COALMINE']
        energy_col = 'Coal*'
    elif df_trade.iloc[0,df_trade.columns.get_loc('Commodity Code')] in [271121, 271111]:
        carrier = 'gas'
        carrier_sources =['OILFIELD']
        energy_col='Natural gas'
    else:
        carrier = 'oil'
        carrier_sources = ['OILWELL']
        energy_col = 'Crude oil'
        
    logger = logging.getLogger(f'vis trade {carrier}')
    
    # drop weird double-france
    ne = ne[~ne.index.isin([249])]
    
    logger.info('Reading opt pickel')
    df_flow_adj = pickle.load(open(params['flowfill_run'][carrier],'rb'))['SIM_ADJ']
    
    fig = plt.figure(figsize=(18,18))
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    gs = gridspec.GridSpec(ncols=4, nrows=14, figure=fig)
    axs = {}
    axs['act'] = fig.add_subplot(gs[0:7,:])
    axs['sim'] = fig.add_subplot(gs[8:,:])
    axs['cax_act_chloro'] = fig.add_axes([0.6,0.5,0.3,0.015])
    axs['cax_sim_chloro'] = fig.add_axes([0.6,0.07,0.3,0.015])
    axs['cax_sim_arrows'] = fig.add_axes([0.125,0.07,0.3,0.015])
        
    ####### do baseline ###
    # prep trade
    
    print (df_trade)
    
    df_trade = df_trade.loc[(~df_trade['Partner ISO'].isna())&(~df_trade['Reporter ISO'].isna())&(df_trade['Year']==2018)&(df_trade['Trade Flow']=='Import')&(df_trade['Reporter ISO']!='WLD')&(df_trade['Partner ISO']!='WLD'),['Reporter ISO','Partner ISO','Qty']]
    df_trade = df_trade.groupby(['Reporter ISO','Partner ISO']).sum().reset_index()
    
    iso2.loc[iso2['iso2']=="'NA",'iso2']='NA'
    
    df_trade = pd.merge(df_trade,iso2[['iso2','iso3']], how='left',left_on='Reporter ISO', right_on='iso3').drop(columns=['iso3']).rename(columns={'iso2':'reporter_iso2'})
    df_trade = pd.merge(df_trade,iso2[['iso2','iso3']], how='left',left_on='Partner ISO', right_on='iso3').drop(columns=['iso3']).rename(columns={'iso2':'partner_iso2'})
    
    # trade_qty in kg, convert to TJ
    df_trade['TJ'] = df_trade['Qty']/1000/params['tperTJ'][carrier]
    df_energy[f'{carrier}_TJ'] = df_energy[energy_col]*41868/1000 #ktoe-> TJ
    print ('energy sorted')
    print (df_energy.sort_values(f'{carrier}_TJ'))
    df_trade['reporter_iso2'] = df_trade['reporter_iso2'].str.replace("'NA",'NA')
    df_trade['partner_iso2'] = df_trade['partner_iso2'].str.replace("'NA",'NA')
    df_trade = df_trade[['reporter_iso2','partner_iso2','TJ']].set_index(['reporter_iso2','partner_iso2']).unstack().rename_axis(['meow','partner_iso2'],axis='columns').droplevel('meow', axis=1)
    
    # clean up energy
    df_energy = df_energy[~df_energy['ISO_A2'].isin(['lo','WORLD'])]
    df_energy['ISO_A2'] = df_energy['ISO_A2'].astype(str)
    df_energy['ISO_A2'] = df_energy['ISO_A2'].str.replace("nan",'NA')
    df_energy['ISO_A22'] = df_energy['ISO_A2']
    df_energy = df_energy[['ISO_A2','ISO_A22',f'{carrier}_TJ']].set_index(['ISO_A2','ISO_A22']).unstack().rename_axis('reporter_iso2').rename_axis(['meow','partner_iso2'],axis='columns').droplevel('meow', axis=1) 
    
    ann_prod_act = pd.Series(np.diag(df_energy), index=[df_energy.index], name='TJ')
    ann_prod_act.index = ann_prod_act.index.get_level_values(0)
    
    #########
    
    if carrier=='coal':
        MIN=100000
        MAX=5000000
        MIN_LW = 1
        MAX_LW = 10
        #df_trade = df_trade[df_trade['TJ']>MIN]
    elif carrier=='gas':
        MIN=1e5
        MAX=1.5e6
        MIN_LW = 1
        MAX_LW = 10
        #df_trade = df_trade[df_trade['TJ']>MIN]
    elif carrier=='oil':
        MIN=1e5
        MAX=3e7
        MIN_LW = 1
        MAX_LW = 10
        #df_trade = df_trade[df_trade['TJ']>MIN]
    
    # reporter = importer
    df_trade = df_trade.stack().reset_index().rename(columns={'partner_iso2':'source','reporter_iso2':'dest',0:'TJ'})
    logger.info('Prepped act trade')
    print (df_trade.sort_values('TJ'))    
    
    df_flow_adj = df_flow_adj.stack()[df_flow_adj.stack()>0].reset_index().rename(columns={'iso2':'source','level_1':'dest',0:'flow'})
    
    df_flow_adj['TJ'] = df_flow_adj['flow']/df_flow_adj['flow'].sum()*ann_prod_act.sum()
    df_flow_adj = df_flow_adj.drop(columns=['flow'])
    
    ann_prod_sim = df_flow_adj[['source','TJ']].groupby('source').sum()
    df_flow_adj = df_flow_adj.loc[df_flow_adj['source']!=df_flow_adj['dest'],:]
    
    logger.info('prepped sim trade')
    print (df_flow_adj)
    
    logger.info('prepped prod act')
    print (ann_prod_act)
    
    logger.info('prepped prod sim')
    print (ann_prod_sim)
    
    print ('sums')
    print (df_trade['TJ'].sum())
    #print (df_energy.sum().sum())
    print (df_flow_adj['TJ'].sum())
    print (ann_prod_sim['TJ'].sum())
    print (ann_prod_act.sum())
    
        
    logger.info('Prepping layout data')
    # filter and join trade data
    
    #ne = ne.drop(index=249)
    ne = pd.merge(ne,ann_prod_act, how='left', left_on='ISO_A2',right_index=True).rename(columns={'TJ':'act_TJ'})
    ne = pd.merge(ne,ann_prod_sim, how='left', left_on='ISO_A2',right_index=True).rename(columns={'TJ':'sim_TJ'})
    
    ne['log10_act_TJ'] = np.log10(ne['act_TJ'])
    #ne['sim_TJ'] = np.log10(ne['sim_TJ'])
    ne.loc[ne['log10_act_TJ']<0,'log10_act_TJ'] = 0
    #ne.loc[ne['sim_TJ']<0,'sim_TJ'] = 0
    
    
    
    ne['logerr'] = np.abs((ne['act_TJ']-ne['sim_TJ'].fillna(0))/ne['act_TJ'])
    print (ne[['log10_act_TJ','act_TJ','sim_TJ','logerr']])
    
    print (ne[['log10_act_TJ','act_TJ','sim_TJ','logerr']].min())
    print (ne[['log10_act_TJ','act_TJ','sim_TJ','logerr']].max())
    print (ne.sort_values('logerr'))
    
    ne.plot(ax=axs['act'], column='log10_act_TJ', missing_kwds={"color": "#c7c7c7"}, edgecolor='white', cmap='viridis', vmin=0, vmax=8)
    ne.plot(ax=axs['sim'], column='logerr', missing_kwds={"color": "#c7c7c7"}, edgecolor='white',cmap='spring_r', vmin=0, vmax=1) 
    
    cmap_v = plt.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=0, vmax=8)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_v), cax=axs['cax_act_chloro'], orientation='horizontal', label='Actual Production [TJ]')
    axs['cax_act_chloro'].set_xticklabels([f'10^{el}' for el in range(9)])
    
    cmap_spring = plt.cm.get_cmap('spring_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_spring), cax=axs['cax_sim_chloro'], orientation='horizontal', label='Simulated Production Error [%]')
    axs['cax_sim_chloro'].set_xticklabels(['0%','20%','40%','60%','80%','100%'])
    
    def get_reppt(geom):
        if geom.type=='MultiPolygon':
            return sorted(list(geom), key=lambda subgeom: subgeom.area)[-1].representative_point()
        elif geom.type=='Polygon':
            return geom.representative_point()
        else:
            print ('ruh roh geom', geom.type)
    
    ne['pt'] = ne['geometry'].apply(lambda geom: get_reppt(geom))
    
    df_trade = pd.merge(df_trade,ne[['pt','ISO_A2']], how='left',left_on='source',right_on='ISO_A2').rename(columns={'pt':'source_pt'}).drop(columns=['ISO_A2'])
    df_trade = pd.merge(df_trade,ne[['pt','ISO_A2']], how='left',left_on='dest',right_on='ISO_A2').rename(columns={'pt':'dest_pt'}).drop(columns=['ISO_A2'])
    df_flow_adj = pd.merge(df_flow_adj,ne[['pt','ISO_A2']], how='left',left_on='source',right_on='ISO_A2').rename(columns={'pt':'source_pt'}).drop(columns=['ISO_A2'])
    df_flow_adj = pd.merge(df_flow_adj,ne[['pt','ISO_A2']], how='left',left_on='dest',right_on='ISO_A2').rename(columns={'pt':'dest_pt'}).drop(columns=['ISO_A2'])
        
    axs['act'].set_xlim([-180,180])
    axs['act'].set_ylim([-60,85])
    axs['sim'].set_xlim([-180,180])
    axs['sim'].set_ylim([-60,85])
    axs['act'].set_xticks([])
    axs['act'].set_yticks([])
    axs['sim'].set_xticks([])
    axs['sim'].set_yticks([])
        
    # green -> good, red -> missing
    df_trade = df_trade.set_index(['source','dest'])
    df_flow_adj = df_flow_adj.set_index(['source','dest'])
    
    df_flow_adj = pd.merge(df_flow_adj, df_trade[['TJ']].rename(columns={'TJ':'R_TJ'}), how='left',left_index=True, right_index=True)
    df_flow_adj['abserr'] = (df_flow_adj['R_TJ'] - df_flow_adj['TJ'])/df_flow_adj['R_TJ']
    
    df_trade = df_trade.reset_index()
    df_flow_adj = df_flow_adj.reset_index()
    
    print (df_trade.sort_values('TJ'))
    print (df_flow_adj.sort_values('TJ'))
    print (len(df_trade.loc[df_trade['TJ']>MIN,:]))
    print (df_flow_adj.sort_values('abserr'))
    
    cmap = plt.cm.get_cmap('PiYG_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axs['cax_sim_arrows'], orientation='horizontal', label='Simulated Trade Error [%]')
    axs['cax_sim_arrows'].set_xticklabels(['0%','20%','40%','60%','80%','100%'])
    
    
    
    for idx, row in df_trade.loc[df_trade['TJ']>MIN,:].sort_values('TJ').iterrows():
        
        print (row['source'],row['dest'],row['TJ'])
        
        axs['act'].annotate("",
                xy=(row['dest_pt'].x, row['dest_pt'].y), 
                xycoords='data',
                xytext=(row['source_pt'].x, row['source_pt'].y), 
                textcoords='data',
                arrowprops=dict(arrowstyle="->", 
                                color="0.5",
                                shrinkA=5, 
                                shrinkB=5,
                                patchA=None, 
                                patchB=None,
                                linewidth=(row['TJ']-MIN)/(MAX-MIN)*(MAX_LW-MIN_LW)+MIN_LW,
                                connectionstyle="arc3,rad=-0.3",
                                ),
                )

    for idx, row in df_flow_adj.loc[df_flow_adj['TJ']>MIN,:].sort_values('TJ').iterrows():
        
        if np.isnan(row['abserr']):
            color="0.5"
        else:
            color = "#{:02x}{:02x}{:02x}".format(*[int(el*255) for el in cmap(row['abserr'])[0:3]])
            
        print (row['source'],row['dest'],row['TJ'])
        
        axs['sim'].annotate("",
                xy=(row['dest_pt'].x, row['dest_pt'].y), 
                xycoords='data',
                xytext=(row['source_pt'].x, row['source_pt'].y), 
                textcoords='data',
                arrowprops=dict(arrowstyle="->", 
                                color=color,
                                shrinkA=5, 
                                shrinkB=5,
                                patchA=None, 
                                patchB=None,
                                linewidth=(row['TJ']-MIN)/(MAX-MIN)*(MAX_LW-MIN_LW)+MIN_LW,
                                connectionstyle="arc3,rad=-0.3",
                                ),
                )

        
        
    logger.info('Done arrows')
    # plt.figimage
    
    # two subplots
    #plt.show()
    
    plt.savefig(os.path.join(os.getcwd(),'results','figures',f'trade_arrows_{carrier}.png'))
    
    return []