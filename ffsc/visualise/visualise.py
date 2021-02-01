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
    df_flow = df_flow.rename(columns={'SOURCE':'source','TARGET':'target'})
    df_flow = df_flow.set_index(['source','target'])
    df_community_edges['source_type'] = df_community_edges['source'].str.split('_').str[0]
    df_community_edges['target_type'] = df_community_edges['target'].str.split('_').str[0]
    df_community_edges = df_community_edges.set_index(['source','target'])
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


def visualise_community_gdfs(params, community_ids, gdfs, ne):
    
    fig, axs = plt.subplots(len(community_ids)//5,min(5,len(community_ids)),figsize=(12*min(5,len(community_ids)),len(community_ids)//5*12))
    axs = axs.flatten()
    
    params['type_style']['ne']['edgecolor']='white'
    
    for ii_c, community_id in enumerate(community_ids):
        ne.plot(ax=axs[ii_c], color='#{:02x}{:02x}{:02x}'.format(*params['vis_colors']['ne']), **params['type_style']['ne'])
        
        extents = []
        
        for dd in gdfs[community_id]:
            dd['gdf'].plot(
                ax=axs[ii_c], 
                color='#{:02x}{:02x}{:02x}'.format(*params['vis_colors'][dd['color_key']]),
                **params['type_style'][dd['type']]
            )
            if len(dd['gdf'])>0:
                extents.append(dd['gdf'].bounds)
            #if dd['type']=='edges':
            #    print ('edges',dd['gdf'])
        
        if len(extents)>0:
            extents = pd.concat(extents)


            minx = extents['minx'].min() if not np.isnan(extents['minx'].min()) else -180
            maxx = extents['maxx'].max() if not np.isnan(extents['maxx'].max()) else 180
            miny = extents['miny'].min() if not np.isnan(extents['miny'].min()) else -90
            maxy = extents['maxy'].max() if not np.isnan(extents['maxy'].max()) else 90
            axs[ii_c].set_xlim([minx, maxx])
            axs[ii_c].set_ylim([miny, maxy])
    plt.savefig(params['path'])

def visualise_communities_do_slice(community_ids, node_slice, edge_slice, params):
    """ return a list of gdf specs"""
    
    gdfs = {}
    for community_id in community_ids:
        gdfs[community_id] = []
        for kk in node_slice['color_key'].unique().tolist() + edge_slice['color_key'].unique().tolist():
            #print (kk,'edges', edge_slice[(edge_slice['color_key']==kk) & (edge_slice['source_comm']==community_id) & (edge_slice['target_comm']==community_id)])
            gdfs[community_id].append(
                {
                    'gdf':gpd.GeoDataFrame(node_slice[(node_slice['color_key']==kk)&(node_slice[params['comm_col']]==community_id)], geometry='geometry'),
                    'color_key':kk,
                    'type':'pt_asset'
                }
            )
            gdfs[community_id].append(
                {
                    
                    'gdf':gpd.GeoDataFrame(edge_slice[(edge_slice['color_key']==kk) & (edge_slice['source_comm']==community_id) & (edge_slice['target_comm']==community_id)], geometry='geometry'),
                    'color_key':kk,
                    'type':'edges'       
                }
            )
            
    return gdfs

def visualise_communities_blobs(df_communities, ne, params):
    
    all_node_types = df_communities[['NODE_TYPES']].explode('NODE_TYPES')['NODE_TYPES'].unique()
    
    
    if 'COALMINE' in all_node_types:
        carrier='coal'
    elif 'LNGTERMINAL' in all_node_types:
        carrier='gas'
    else:
        carrier='oil'
        
    logger = logging.getLogger(f'Visualise community blobs {carrier}')
    logger.info('loading geoetries')
    df_communities['geometry'] = df_communities['geometry'].apply(wkt.loads)
    df_communities['convex_hull'] = df_communities['geometry'].apply(lambda el: el.convex_hull)
    gdf = gpd.GeoDataFrame(df_communities, geometry='convex_hull')
    
    fig, ax = plt.subplots(1,1,figsize=(48,36))
    ne.plot(ax=ax, color='#{:02x}{:02x}{:02x}'.format(*params['vis_colors']['ne']), **params['type_style']['ne'])
    gdf[(gdf['supply']==False) & (gdf['demand']==False)].boundary.plot(ax=ax, color='#{:02x}{:02x}{:02x}'.format(*params['vis_colors']['SHIPPINGROUTE'])) # transmission
    gdf[(gdf['supply']==True) & (gdf['demand']==False)].boundary.plot(ax=ax, color='#{:02x}{:02x}{:02x}'.format(*params['vis_colors']['COALMINE'])) # transmission
    gdf[(gdf['supply']==False) & (gdf['demand']==True)].boundary.plot(ax=ax, color='#{:02x}{:02x}{:02x}'.format(*params['vis_colors']['OILFIELD'])) # transmission
    gdf[(gdf['supply']==True) & (gdf['demand']==True)].boundary.plot(ax=ax, color='#{:02x}{:02x}{:02x}'.format(*params['vis_colors']['REFINERY'])) # transmission
        
    plt.savefig(os.path.join(os.getcwd(),'results','figures',f'community_blobs_{carrier}.png'))   
    return []
    
def visualise_communities_wrapper(community_ids, df_nodes, df_edges, params, ne):
    gdfs = visualise_communities_do_slice(
        community_ids=community_ids, 
        node_slice = df_nodes[df_nodes[params["comm_col"]].isin(community_ids)], 
        edge_slice = df_edges[df_edges['source_comm'].isin(community_ids) | df_edges['target_comm'].isin(community_ids)], 
        params=params
    )
    visualise_community_gdfs(params, community_ids, gdfs, ne)

def visualise_communities_detail(params, df_nodes, df_edges, df_communities, ne):
    if 'COALMINE' in df_nodes['NODETYPE'].unique():
        carrier='coal'
    elif 'LNGTERMINAL' in df_nodes['NODETYPE'].unique():
        carrier='gas'
    else:
        carrier='oil'
        
    logger=logging.getLogger('Visualise communities '+carrier)
    comm_col = f'comm_{params["community_levels"][carrier]-1}'
    params['comm_col']=comm_col
    
    logger.info('Loading geometries')
    df_nodes['geometry'] = df_nodes['geometry'].apply(wkt.loads)
    df_edges['geometry'] = df_edges['geometry'].apply(wkt.loads)
    df_communities['geometry'] = df_communities['geometry'].apply(wkt.loads)
    
    
    df_edges['source_type'] = df_edges['source'].str.split('_').str[0]
    df_edges['target_type'] = df_edges['target'].str.split('_').str[0]
    
    df_edges['color_key'] = 'FINALMILE'
    for kk in ['RAILWAY','PIPELINE','SHIPPINGROUTE']:
        df_edges.loc[((df_edges['source_type']==kk) | (df_edges['target_type']==kk)),'color_key'] = kk
        
    #print ('df_Edges')
    #print(df_edges)
        
    df_nodes = df_nodes[~df_nodes['NODETYPE'].isin(['RAILWAY','PIPELINE','SHIPPINGROUTE'])]
    df_nodes['color_key'] = df_nodes['NODETYPE']
    
    df_communities = df_communities.sort_values('N_NODES')
    
    ### what to vis
    # top largest
    logger.info(f'Doing {params["vis_N_communities"]} largest')
    community_ids = df_communities.iloc[-1*params['vis_N_communities']:].index.values
    params['path'] = os.path.join(os.getcwd(),'results','figures',f'communities_{carrier}_{params["vis_N_communities"]}_largest.png')
    visualise_communities_wrapper(community_ids, df_nodes, df_edges, params, ne)
    
    # smallest
    logger.info(f'Doing {params["vis_N_communities"]} smallest')
    community_ids = df_communities.iloc[:params['vis_N_communities']].index.values
    params['path'] = os.path.join(os.getcwd(),'results','figures',f'communities_{carrier}_{params["vis_N_communities"]}_smallest.png')
    visualise_communities_wrapper(community_ids, df_nodes, df_edges, params, ne)
    
    # top supply
    logger.info(f'Doing {params["vis_N_communities"]} largest - supply')
    community_ids = df_communities.loc[(df_communities['supply']==True)&(df_communities['demand']==False),:].iloc[-1*params['vis_N_communities']:].index.values
    params['path'] = os.path.join(os.getcwd(),'results','figures',f'communities_{carrier}_{params["vis_N_communities"]}_largest_supply.png')
    visualise_communities_wrapper(community_ids, df_nodes, df_edges, params, ne)
    
    # smallest supply
    logger.info(f'Doing {params["vis_N_communities"]} smallest - supply')
    community_ids = df_communities.loc[(df_communities['supply']==True)&(df_communities['demand']==False),:].iloc[:params['vis_N_communities']].index.values
    params['path'] = os.path.join(os.getcwd(),'results','figures',f'communities_{carrier}_{params["vis_N_communities"]}_smallest_supply.png')
    visualise_communities_wrapper(community_ids, df_nodes, df_edges, params, ne)
    
    # top demand
    logger.info(f'Doing {params["vis_N_communities"]} largest - demand')
    community_ids = df_communities.loc[(df_communities['supply']==False)&(df_communities['demand']==True),:].iloc[-1*params['vis_N_communities']:].index.values
    params['path'] = os.path.join(os.getcwd(),'results','figures',f'communities_{carrier}_{params["vis_N_communities"]}_largest_demand.png')
    visualise_communities_wrapper(community_ids, df_nodes, df_edges, params, ne)
    
    # smallest_ demand
    logger.info(f'Doing {params["vis_N_communities"]} smallest - demand')
    community_ids = df_communities.loc[(df_communities['supply']==False)&(df_communities['demand']==True),:].iloc[:params['vis_N_communities']].index.values
    params['path'] = os.path.join(os.getcwd(),'results','figures',f'communities_{carrier}_{params["vis_N_communities"]}_smallest_demand.png')
    visualise_communities_wrapper(community_ids, df_nodes, df_edges, params, ne)
    
    # top transmission
    logger.info(f'Doing {params["vis_N_communities"]} largest - transmission')
    community_ids = df_communities.loc[(df_communities['supply']==False)&(df_communities['demand']==False),:].iloc[-1*params['vis_N_communities']:].index.values
    params['path'] = os.path.join(os.getcwd(),'results','figures',f'communities_{carrier}_{params["vis_N_communities"]}_largest_transmission.png')
    visualise_communities_wrapper(community_ids, df_nodes, df_edges, params, ne)
    
    # smallest transmission
    logger.info(f'Doing {params["vis_N_communities"]} smallest - transmission')
    community_ids = df_communities.loc[(df_communities['supply']==False)&(df_communities['demand']==False),:].iloc[:params['vis_N_communities']].index.values
    params['path'] = os.path.join(os.getcwd(),'results','figures',f'communities_{carrier}_{params["vis_N_communities"]}_smallest_transmission.png')
    visualise_communities_wrapper(community_ids, df_nodes, df_edges, params, ne)
    
    # top supple+demand
    logger.info(f'Doing {params["vis_N_communities"]} largest - supply+demand')
    community_ids = df_communities.loc[(df_communities['supply']==True)&(df_communities['demand']==True),:].iloc[-1*params['vis_N_communities']:].index.values
    params['path'] = os.path.join(os.getcwd(),'results','figures',f'communities_{carrier}_{params["vis_N_communities"]}_largest_transmission.png')
    visualise_communities_wrapper(community_ids, df_nodes, df_edges, params, ne)
    
    # smallest supply+demand
    logger.info(f'Doing {params["vis_N_communities"]} smallest - supply+demand')
    community_ids = df_communities.loc[(df_communities['supply']==True)&(df_communities['demand']==True),:].iloc[:params['vis_N_communities']].index.values
    params['path'] = os.path.join(os.getcwd(),'results','figures',f'communities_{carrier}_{params["vis_N_communities"]}_smallest_transmission.png')
    visualise_communities_wrapper(community_ids, df_nodes, df_edges, params, ne)
    
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
    
    
    G = nx.DiGraph()
    G.add_edges_from([(r[0],r[1],{'flow':r[2], 'flow_inv':r[3]}) for r in df_flow[['SOURCE','TARGET','flow','inv_flow']].values.tolist()])
    
    MI = pd.MultiIndex.from_product([iso2['iso2'].unique().tolist(),iso2['iso2'].unique().tolist()], names=('source', 'target'))
    all_flows = pd.DataFrame(index=MI)
    
    all_flows['flow'] = 0
    
    sources = df_flow.loc[df_flow['TARGET'].str.split('_').str[0].isin(['COALMINE']) & (df_flow['flow']>0),'TARGET'].values.tolist()
    #sources = df_community_iso2.loc[df_community_iso2['NODE'].str.split('_').str[0].isin(carrier_sources),'NODE'].values.tolist()
    #print (sources)
    logger.info(f'Number of sources {len(sources)}')
    
    
    df_community_iso2 = df_community_iso2.set_index('NODE')
    df_flow = df_flow.set_index(['SOURCE','TARGET'])
    
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
    

def visualise_trade(iso2, ne, df_energy, df_trade, df_flow_adj):
    
    if df_trade.iloc[0,df_trade.columns.get_loc('Commodity Code')]==2701:
        carrier='coal'
        carrier_sources=['COALMINE']
    elif df_trade.iloc[0,df_trade.columns.get_loc('Commodity Code')] in [271121, 271111]:
        carrier = 'gas'
        carrier_sources =['OILFIELD']
    else:
        carrier = 'oil'
        carrier_sources = ['OILWELL']
        
    logger = logging.getLogger(f'vis trade {carrier}')
    
    fig, axs = plt.subplots(2,1,figsize=(48,48))
    
    ne.plot(ax=axs[0], color='#c7c7c7',edgecolor='white')
    ne.plot(ax=axs[1], color='#c7c7c7',edgecolor='white')
    
    logger.info('Prepping trade data')
    # filter and join trade data
    
    df_trade = df_trade.loc[(~df_trade['Partner ISO'].isna())&(~df_trade['Reporter ISO'].isna())&(df_trade['Year']==2018)&(df_trade['Trade Flow']=='Import')&(df_trade['Reporter ISO']!='WLD')&(df_trade['Partner ISO']!='WLD'),['Reporter ISO','Partner ISO','Qty']]
    
    iso2.loc[iso2['iso2']=="'NA",'iso2']='NA'
    
    df_trade = pd.merge(df_trade,iso2[['iso2','iso3']], how='left',left_on='Reporter ISO', right_on='iso3').drop(columns=['iso3']).rename(columns={'iso2':'reporter_iso2'})
    df_trade = pd.merge(df_trade,iso2[['iso2','iso3']], how='left',left_on='Partner ISO', right_on='iso3').drop(columns=['iso3']).rename(columns={'iso2':'partner_iso2'})
    
    ne['pt'] = ne['geometry'].representative_point()
    ne = ne.rename(columns={'ISO_A2':'iso2'})
    ne['iso2'] = ne['iso2'].astype(str)
    
    df_trade = pd.merge(df_trade, ne[['iso2','pt']], how='left',left_on='reporter_iso2',right_on='iso2').rename(columns={'pt':'pt_reporter'})
    df_trade = pd.merge(df_trade, ne[['iso2','pt']], how='left',left_on='partner_iso2',right_on='iso2').rename(columns={'pt':'pt_partner'})
    
    if carrier=='coal':
        MIN=100e6
        MAX=100e9
        MIN_LW = 1
        MAX_LW = 25
        df_trade = df_trade[df_trade['Qty']>MIN]
    
    logger.info('Prepped df')
    print (df_trade)
    
    # filter trade data
    
    for idx, row in df_trade.iterrows():
        
        axs[0].annotate("",
                xy=(row['pt_reporter'].x, row['pt_reporter'].y), 
                xycoords='data',
                xytext=(row['pt_partner'].x, row['pt_partner'].y), 
                textcoords='data',
                arrowprops=dict(arrowstyle="->", 
                                color="0.5",
                                shrinkA=5, 
                                shrinkB=5,
                                patchA=None, 
                                patchB=None,
                                linewidth=(row['Qty']-MIN)/(MAX-MIN)*(MAX_LW-MIN_LW)+MIN_LW,
                                connectionstyle="arc3,rad=-0.3",
                                ),
                )
        
        
    logger.info('doing flow solution')
    
    # rescale flow data
    df_flow_adj['flow'] = df_flow_adj['flow']/df_flow_adj['flow'].sum() * df_trade['Qty'].sum()
    df_flow_adj = df_flow_adj.reset_index()
    df_flow_adj = df_flow_adj[df_flow_adj['flow']>MIN]
    df_flow_adj.loc[df_flow_adj['source']=="'NA",'source']='NA'
    df_flow_adj.loc[df_flow_adj['target']=="'NA",'target']='NA'
    
    df_flow_adj = pd.merge(df_flow_adj, ne[['iso2','pt']], how='left',left_on='source',right_on='iso2').rename(columns={'pt':'pt_source'})
    df_flow_adj = pd.merge(df_flow_adj, ne[['iso2','pt']], how='left',left_on='target',right_on='iso2').rename(columns={'pt':'pt_target'})
    
    print('flow minmax',df_flow_adj['flow'].min(), df_flow_adj['flow'].max())
    print (df_flow_adj)
    
    for idx, row in df_flow_adj.iterrows():
        axs[1].annotate("",
                xy=(row['pt_source'].x, row['pt_source'].y), 
                xycoords='data',
                xytext=(row['pt_target'].x, row['pt_target'].y), 
                textcoords='data',
                arrowprops=dict(arrowstyle="->", 
                                color="0.5",
                                shrinkA=5, 
                                shrinkB=5,
                                patchA=None, 
                                patchB=None,
                                linewidth=(row['flow']-MIN)/(MAX-MIN)*(MAX_LW-MIN_LW)+MIN_LW,
                                connectionstyle="arc3,rad=-0.3",
                                ),
                )
        
        
    logger.info('Done arrows')
    # plt.figimage
    
    # two subplots
    #plt.show()
    
    plt.savefig(os.path.join(os.getcwd(),'results','figures',f'trade_{carrier}.png'))
    
    return []