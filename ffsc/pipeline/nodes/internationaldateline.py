from shapely import geometry, affinity

import logging, sys, json, time
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
from itertools import chain

import pandas as pd
from shapely import geometry
from tqdm import tqdm
tqdm.pandas()

from shapely import wkt, geometry, affinity, ops

from ffsc.pipeline.nodes.utils import V_inv

def connect_IDL(df):
    logger = logging.getLogger('shippingroutes_connect_IDL')
    # do an affinity transform, then get nearest points, then reverse the affine transform, then V_inv to get the distance
    
    logger.info('intersecting boxes and taking affine transform')
    pos_box = geometry.box(178,-85, 180,85)
    neg_box = geometry.box(-180,-85,-178,85)
    
    
    df['pos_intersects'] = df['geometry'].apply(lambda el: wkt.loads(el).intersects(pos_box))
    df['neg_intersects'] = df['geometry'].apply(lambda el: wkt.loads(el).intersects(neg_box))
    
    df_pos = df[df['pos_intersects']]
    df_neg = df[df['neg_intersects']]
    
    DT = [1,0,0,1,-360,0]
    
    df_pos['geometry'] = df_pos['geometry'].apply(lambda el: affinity.affine_transform(wkt.loads(el),DT))
    df_neg['geometry'] = df_neg['geometry'].apply(lambda el: wkt.loads(el))
    
    DT = [1,0,0,1,360,0]
    
    logger.info('Getting all nearest Points')
    records = []
    for idx_pos, row_pos in tqdm(df_pos.iterrows()):
        for idx_neg, row_neg in df_neg.iterrows():
            pt_pos, pt_neg = ops.nearest_points(row_pos['geometry'], row_neg['geometry'])
            records.append({
                'pt_pos':affinity.affine_transform(pt_pos,DT),
                'pt_neg':pt_neg,
                'id_pos':row_pos['unique_id'],
                'id_neg':row_neg['unique_id']
            })
            
    df_intersection = pd.DataFrame.from_records(records)
    
    logger.info('Getting distance between points')
    df_intersection['DISTANCE'] = df_intersection.progress_apply(lambda row: V_inv((row['pt_pos'].y, row['pt_pos'].x),(row['pt_neg'].y, row['pt_neg'].x))[0]*1000, axis=1) #m
    df_intersection = df_intersection.rename(columns={'id_pos':'START','id_neg':'END'})
    df_intersection = df_intersection[df_intersection['DISTANCE']<10000]
    df_intersection['PT_START'] = df_intersection['pt_pos'].apply(lambda el: el.wkt)
    df_intersection['PT_END'] = df_intersection['pt_neg'].apply(lambda el: el.wkt)
    
    
    return df_intersection[['START','END','PT_START','PT_END','DISTANCE']]
            
    