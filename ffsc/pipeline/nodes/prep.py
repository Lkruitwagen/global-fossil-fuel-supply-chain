import logging, sys, json, time
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
from itertools import chain

import pandas as pd
from shapely import geometry
from tqdm import tqdm
tqdm.pandas()

from shapely import wkt

from ffsc.pipeline.nodes.mp_sjoin_dissolve import *
from ffsc.pipeline.nodes.mp_utmbuffer import *

N_WORKERS=12

def selfjoin_and_dissolve(df):
    asset_name = df.iloc[0]['unique_id'].split('_')[0]
    logger = logging.getLogger('prep_selfjoin'+'_'+asset_name)
    
    #print ('df')
    #print (df)
    logger.info('retaining unique_ids')
    df['features'] = df.progress_apply(lambda row: json.dumps({**json.loads(row['features']),**{'orig_idx':row['unique_id']}}), axis=1)
    
    logger.info('unnest geometries')
    def unnest_multils(geom):
        if geom.split(' ')[0]=='MULTILINESTRING':
            return [subgeom.wkt for subgeom in list(wkt.loads(geom))]
        else:
            return geom
        
    def break_ls(ls):
        return [geometry.LineString([cc1, cc2]) for cc1, cc2 in zip(ls.coords[0:-1], ls.coords[1:])]

    def nest_break_ls(geom):
        if geom.type=='MultiLineString':
            return list(chain.from_iterable([break_ls(ls) for ls in list(geom)]))
        elif geom.type=='LineString':
            return break_ls(geom)

    df['geometry'] = df['geometry'].progress_apply(unnest_multils)

    logger.info('exploding MultiLineStrings')
    df = df.explode('geometry')
    
    # explode linestrings before selfjoin - more efficient STRTree
    logger.info('Now explode linestring parts')
    df['geometry'] = df['geometry'].progress_apply(lambda el: wkt.loads(el))
    
    df['geometry'] = df['geometry'].progress_apply(lambda el: nest_break_ls(el))
    
    logger.info('Now explode and turn back to str')
    df = df.explode('geometry')
    
    df['geometry'] = df['geometry'].progress_apply(lambda el:el.wkt)
    
    
    
    # sjoin to self
    intersection_df = pd.DataFrame(
                            mp_sjoin_dissolve(
                                df_left=df, 
                                df_right=df, 
                                left_geom_column='geometry', 
                                right_geom_column='geometry', 
                                N_workers=N_WORKERS, 
                                logger=logger,
                                include_geometry=True),
                            columns=['L_idx','R_idx','intersects', 'intersection_pts'])   
    
    # only retain where true
    #print ('intersectino_df',intersection_df)
    intersection_df = intersection_df[intersection_df['intersects']==True]    
    
    # get intersection between geom columns
    #print ('intersectino_df',intersection_df)
    
    flatten = lambda t: [item for sublist in t for item in sublist]
    # split_pts
    #print (intersection_df.groupby('L_idx')['intersection_pts'].apply(lambda el: list(set(flatten(el)))).reset_index())
    
    # merge onto main df
    logger.info('merging onto main df')

    df = pd.merge(
        df, 
        intersection_df.groupby('L_idx')['intersection_pts'].apply(lambda el: list(set(flatten(el)))).reset_index(),
        how='left',
        left_index=True,
        right_on='L_idx'
    ).reset_index()
    
    # split each row of main df
    logger.info('splitting linestrings')
    #print (df)
    #print (geometry.MultiPoint(df['intersection_pts'][0]))
    def maybe_split_geom(row):
        if not np.isnan(row['intersection_pts']).any():
            if len(row['intersection_pts'])>0:
                return list(ops.split(wkt.loads(row['geometry']), geometry.MultiPoint(row['intersection_pts'])))
            else:
                return [wkt.loads(row['geometry'])]
        else:
            return [wkt.loads(row['geometry'])]
            
    
    df['geometry'] = df.progress_apply(lambda row: maybe_split_geom(row), axis=1)
    
    #print (df)
    #print (df['geometry'].str.len())
    logger.info('exploding splitted geometry and cleaning up')
    df = df.explode('geometry')
    
    
    
    logger.info('write geometry to str')
    df['geometry'] = df['geometry'].progress_apply(lambda el: el.wkt)
    
    df = df.drop(columns=['intersection_pts','L_idx','index']).reset_index()
    
    df['unique_id'] = asset_name+'_'+df['index'].astype(str)
    
    df = df.drop(columns=['index'])   
    
    return df
    
def buffer_and_split(df, parameters):
    asset_name = df.iloc[0]['unique_id'].split('_')[0]
    logger = logging.getLogger('prep_buffer'+'_'+asset_name)
    
    logger.info(f'Getting geometries buffered by {parameters["BUFFER"][asset_name]}m')
    df['buffer_geom'] = mp_buffer(df,parameters['BUFFER'][asset_name],N_WORKERS)
    
    logger.info(f'Mapping buffer geom to str')
    df['buffer_geom'] = df['buffer_geom'].apply(lambda el: el.wkt)
    
    return df