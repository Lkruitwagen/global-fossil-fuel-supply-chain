import logging, sys, json, time
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
from itertools import chain

import pandas as pd
from shapely import geometry, wkt, ops
from tqdm import tqdm
tqdm.pandas()

from ffsc.pipeline.nodes.utils import V_inv
    
def format_pt(tup, linear_name):
    return f'{linear_name}_PT_{round(tup[0],4):.4f}_{round(tup[1],4):.4f}'
    
def explode_edges_pipelines(df,edge_df1, edge_df2, edge_df3, edge_df4, edge_df5, edge_df6, edge_df7, edge_df8, edge_df9, edge_df10):
    logger = logging.getLogger('explode_edges'+'_'+'pipelines')
    
    # attach the pipeline geoms, get nearest point
    
    edge_df_pipelines_other = pd.concat([edge_df1, edge_df2, edge_df3, edge_df4, edge_df5, edge_df6, edge_df7,edge_df8])
    edge_df_other_pipelines = pd.concat([edge_df9,edge_df10])
    
    node_df, edge_df, keep_df,edge_df_pipelines_other, edge_df_other_pipelines = explode_edges_linear(df, edge_df_pipelines_other, edge_df_other_pipelines, 'PIPELINE',logger)
    
    return node_df, edge_df, keep_df,edge_df_pipelines_other, edge_df_other_pipelines
    
    
def explode_edges_railways(df, edge_df1, edge_df2, edge_df3, edge_df4, edge_df5, edge_df6):
    logger = logging.getLogger('explode_edges'+'_'+'railways')
    
    edge_df_railways_other = pd.concat([edge_df1, edge_df2, edge_df3, edge_df4, edge_df5])
    edge_df_other_railways = edge_df6
    
    node_df, edge_df, keep_df,edge_df_railways_other, edge_df_other_railways = explode_edges_linear(df, edge_df_railways_other, edge_df_other_railways, 'RAILWAY',logger)
    
    return node_df, edge_df, keep_df,edge_df_railways_other, edge_df_other_railways
    
def explode_edges_shippingroutes(df, edge_df1, edge_df2, IDL_edges):
    logger = logging.getLogger('explode_edges'+'_'+'shippingroutes')
    
    logger.info(f'Concatenating edge dataframes')
    edge_df_shippingroutes_other = pd.concat([edge_df1, edge_df2, IDL_edges.rename(columns={'PT_END':'NEAREST_PT'})])
    edge_df_other_shippingroutes = IDL_edges.rename(columns={'PT_START':'NEAREST_PT'})
    
    node_df, edge_df, keep_df,edge_df_shippingroutes_other, edge_df_other_shippingroutes = explode_edges_linear(df, edge_df_shippingroutes_other, edge_df_other_shippingroutes, 'SHIPPINGROUTE',logger)
    
    logger.info(f'Adding IDL edges')   
    IDL_edges['START'] = IDL_edges['PT_START'].apply(lambda el: format_pt((wkt.loads(el).x, wkt.loads(el).y), 'SHIPPINGROUTE'))
    IDL_edges['END'] = IDL_edges['PT_END'].apply(lambda el: format_pt((wkt.loads(el).x, wkt.loads(el).y), 'SHIPPINGROUTE'))

    edge_df = pd.concat([edge_df, IDL_edges[['START','END','DISTANCE']]])
    
    return node_df, edge_df, keep_df,edge_df_shippingroutes_other, edge_df_other_shippingroutes
    
def explode_edges_linear(df, edge_df_linear_other, edge_df_other_linear, linear_name, logger):
    
    # get only linear
    logger.info('Getting only linear assets')
    edge_df_linear_other = edge_df_linear_other[edge_df_linear_other['START'].str.split('_').str[0]==linear_name]
    edge_df_other_linear = edge_df_other_linear[edge_df_other_linear['END'].str.split('_').str[0]==linear_name]
        
    # groupby pipeline_unique_id, cast nearest_points to multipoint
    logger.info('Grouping by ID and getting multipoints')
    nearest_pt_linear_other = edge_df_linear_other.groupby('START')['NEAREST_PT'].apply(list).reset_index().rename(columns={'START':'unique_id'})
    nearest_pt_other_linear = edge_df_other_linear.groupby('END')['NEAREST_PT'].apply(list).reset_index().rename(columns={'END':'unique_id'})
    nearest_pt_df = pd.merge(nearest_pt_linear_other,nearest_pt_other_linear, on='unique_id', how='outer')

    nearest_pt_df['NEAREST_PT_x'] = nearest_pt_df['NEAREST_PT_x'].apply(lambda d: d if isinstance(d, list) else [])
    nearest_pt_df['NEAREST_PT_y'] = nearest_pt_df['NEAREST_PT_y'].apply(lambda d: d if isinstance(d, list) else [])

    nearest_pt_df['NEAREST_PTS'] = nearest_pt_df['NEAREST_PT_x'] + nearest_pt_df['NEAREST_PT_y']
    #print (nearest_pt_df) #good!
    # df join groupby with multipoint
    df = pd.merge(df,nearest_pt_df.drop(columns=['NEAREST_PT_x','NEAREST_PT_y']), how='left',on='unique_id')
    
    # cast to geometries
    logger.info(f'Casting to geometries')
    df['geometry'] = df['geometry'].progress_apply(lambda el: wkt.loads(el))
    df['NEAREST_PTS'] = df['NEAREST_PTS'].apply(lambda d: d if isinstance(d, list) else [])
    #print (df['NEAREST_PTS'])
    df['multipoint'] = df['NEAREST_PTS'].progress_apply(lambda el: geometry.MultiPoint([wkt.loads(pt) for pt in el])) # dies on NAN
    
    # Keep the multipoints
    logger.info('Getting the multipoint keep nodes')
    keeppoints = chain.from_iterable([list(multipoint) for multipoint in df['multipoint'].values.tolist()]) # point objects
    keeppoints = [f'{linear_name}_PT_{round(pt.x,4):.4f}_{round(pt.y,4):.4f}' for pt in keeppoints]
    keep_df = pd.DataFrame(keeppoints,columns=['KEEP_NODES'])
    #print ('keepnodes',keep_df)
    #print (df['multipoint'].apply(lambda el: el.type).unique())
    
    # ops.split
    logger.info(f'Calling ops.split on geometries with multipoints')
    df['geometry'] = df.progress_apply(lambda row: list(ops.split(row['geometry'],row['multipoint'])) if not row['multipoint'].is_empty else [row['geometry']], axis=1)
    
    # explode
    logger.info(f'Exploding split geometries')
    df = df.explode('geometry')
    
    # get all pipeline-pipeline edges
    logger.info('getting coordinate sequences')
    all_coord_seqs = df['geometry'].progress_apply(lambda el: list(el.coords)).values.tolist()
    
    edges = []
    logger.info(f'Coordinate sequences to edge lists')
    for seq in tqdm(all_coord_seqs):
        edges+= [
            {
            'START':format_pt(cc1, linear_name),
            'END':format_pt(cc2, linear_name),
            'DISTANCE':V_inv((cc1[1],cc1[0]),(cc2[1], cc2[0]))[0]*1000 #m
            } 
            for cc1, cc2 in zip(seq[:-1],seq[1:])
        ]
        
    edge_df = pd.DataFrame.from_records(edges) 
    
    # get all pipeline nodes
    node_df = pd.DataFrame(list(set(edge_df['START'].unique().tolist()+edge_df['END'].unique().tolist())), columns=['NODES'])
    
    logger.info('Edge formatting')
    edge_df_linear_other['START'] = edge_df_linear_other['NEAREST_PT'].progress_apply(lambda el: format_pt((wkt.loads(el).x, wkt.loads(el).y), linear_name))
    edge_df_other_linear['END']  = edge_df_other_linear['NEAREST_PT'].progress_apply(lambda el: format_pt((wkt.loads(el).x, wkt.loads(el).y), linear_name))
    
    return node_df, edge_df, keep_df, edge_df_linear_other, edge_df_other_linear
    
    
def drop_linear(df):
    linear_assets = ['PIPELINE','RAILWAY','SHIPPINGROUTE']
    
    df = df[~df['START'].str.split('_').str[0].isin(linear_assets)]
    df = df[~df['END'].str.split('_').str[0].isin(linear_assets)]
    
    return df

