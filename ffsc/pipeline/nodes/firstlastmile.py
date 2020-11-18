import logging, sys, json, time, os
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import scipy.spatial
from shapely import geometry, ops
from tqdm import tqdm
tqdm.pandas()
from pandas.core.common import SettingWithCopyWarning

from ffsc.pipeline.nodes.mp_sjoin_mindist import *
from ffsc.pipeline.nodes.mp_utmbuffer import *

N_WORKERS=12


def firstmile_edge(asset_df, linear_edge_df, port_df, city_df, linear_df):
    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=SettingWithCopyWarning)
    # assets, existing_edges, closest port, city, [pipeline/railway]
    asset_name = asset_df.iloc[0]['unique_id'].split('_')[0]
    logger = logging.getLogger('flmile'+'_'+asset_name)
    
    logger.info(f'Start from {len(asset_df)} {asset_name}, eliminate exsting matches')
    asset_df = asset_df.loc[~asset_df['unique_id'].isin(linear_edge_df['END'].values),:]
    
    logger.info(f'multiprocess Buffering {len(asset_df)} {asset_name} to a max distance')
    asset_df['buffer_100km'] = mp_buffer(asset_df,100000,6)
    asset_df['buffer_100km'] = asset_df['buffer_100km'].progress_apply(lambda el: el.wkt)
    
    combined_target_df = pd.concat([port_df, city_df, linear_df])
    
    logger.info(f'Finding closest matches for {len(asset_df)} {asset_name} sources in {len(combined_target_df)} assets')
    
    intersection_df = pd.DataFrame(
                            mp_sjoin_mindist(
                                df_linear=combined_target_df, 
                                df_buffered=asset_df, 
                                left_geom_column='geometry', 
                                right_geom_column='geometry', 
                                right_buffer_column='buffer_100km',
                                N_workers=N_WORKERS, 
                                logger=logger,
                                include_min_dist=True),
                            columns=['L_idx','R_idx','intersects','NEAREST_PT', 'DISTANCE'])  

    intersection_df = intersection_df[intersection_df['intersects']==True]
    intersection_df = intersection_df.sort_values("DISTANCE").groupby("L_idx", as_index=False).first()
    intersection_df['START'] = asset_df.iloc[intersection_df['L_idx'].values,asset_df.columns.get_loc('unique_id')].values
    intersection_df['END'] = combined_target_df.iloc[intersection_df['R_idx'].values,combined_target_df.columns.get_loc('unique_id')].values
    
    return intersection_df[['START','END','NEAREST_PT','DISTANCE']]


def powerstations_lastmile(df_powerstations, df_edges_pipelines, df_edges_railways, df_railways, df_pipelines, df_ports, df_cities):
    # powerstations to the closest lin asset, port, or city
    logger = logging.getLogger('flmile'+'_'+'POWERSTATIONS')
    
    logger.info('un-str features')
    df_powerstations['features'] = df_powerstations['features'].progress_apply(json.loads)
    df_powerstations['fuel1'] = df_powerstations['features'].progress_apply(lambda el: el['fuel1'])
    df_powerstations['fuel2'] = df_powerstations['features'].progress_apply(lambda el: el['fuel2'])
    
    logger.info('remove already-matched assets')
    df_powerstations = df_powerstations.loc[~df_powerstations['unique_id'].isin(df_edges_pipelines['END'].values),:]
    df_powerstations = df_powerstations.loc[~df_powerstations['unique_id'].isin(df_edges_railways['END'].values),:]
    
    logger.info(f'multiprocess Buffering powerstations to a max distance')
    df_powerstations['buffer_100km'] = mp_buffer(df_powerstations,100000,6)
    df_powerstations['buffer_100km'] = df_powerstations['buffer_100km'].progress_apply(lambda el: el.wkt)
    
    logger.info(f'match remaining {len(df_powerstations)} powerstations')
    
    # get closest railway, port, or city
    logger.info('doing Coal')
    source_df = df_powerstations.loc[((df_powerstations['fuel1']=='Coal') | ((df_powerstations['fuel1']=='Cogeneration') & (df_powerstations['fuel2']=='Coal'))),:]
    combined_target_df = pd.concat([df_railways,df_ports,df_cities])
    
    
    coal_intersection = pd.DataFrame(
                            mp_sjoin_mindist(
                                df_linear=combined_target_df, 
                                df_buffered=source_df, 
                                left_geom_column='geometry', 
                                right_geom_column='geometry', 
                                right_buffer_column='buffer_100km',
                                N_workers=N_WORKERS, 
                                logger=logger,
                                include_min_dist=True),
                            columns=['L_idx','R_idx','intersects','NEAREST_PT', 'DISTANCE'])  
    
    coal_intersection = coal_intersection[coal_intersection['intersects']==True]

    
    coal_intersection = coal_intersection.sort_values("DISTANCE").groupby("L_idx", as_index=False).first()
    coal_intersection['END'] = source_df.iloc[coal_intersection['L_idx'].values,source_df.columns.get_loc('unique_id')].values
    coal_intersection['START'] = combined_target_df.iloc[coal_intersection['R_idx'].values,combined_target_df.columns.get_loc('unique_id')].values
    
    logger.info('doing Oil')
    source_df = df_powerstations.loc[((df_powerstations['fuel1']=='Oil') | ((df_powerstations['fuel1']=='Cogeneration') & (df_powerstations['fuel2']=='Oil'))),:]
    combined_target_df = pd.concat([df_pipelines,df_ports,df_cities])
    
    oil_intersection = pd.DataFrame(
                            mp_sjoin_mindist(
                                df_linear=combined_target_df, 
                                df_buffered=source_df, 
                                left_geom_column='geometry', 
                                right_geom_column='geometry', 
                                right_buffer_column='buffer_100km',
                                N_workers=N_WORKERS, 
                                logger=logger,
                                include_min_dist=True),
                            columns=['L_idx','R_idx','intersects', 'NEAREST_PT','DISTANCE'])  
    
    oil_intersection = oil_intersection[oil_intersection['intersects']==True]
 
    
    oil_intersection = oil_intersection.sort_values("DISTANCE").groupby("L_idx", as_index=False).first()
    oil_intersection['END'] = source_df.iloc[oil_intersection['L_idx'].values,source_df.columns.get_loc('unique_id')].values
    oil_intersection['START'] = combined_target_df.iloc[oil_intersection['R_idx'].values,combined_target_df.columns.get_loc('unique_id')].values
    
    logger.info('doing gas')
    source_df = df_powerstations.loc[((df_powerstations['fuel1']=='Gas') | ((df_powerstations['fuel1']=='Cogeneration') & (df_powerstations['fuel2']=='Gas'))),:]
    
    gas_intersection = pd.DataFrame(
                            mp_sjoin_mindist(
                                df_linear=combined_target_df, 
                                df_buffered=source_df, 
                                left_geom_column='geometry', 
                                right_geom_column='geometry', 
                                right_buffer_column='buffer_100km',
                                N_workers=N_WORKERS, 
                                logger=logger,
                                include_min_dist=True),
                            columns=['L_idx','R_idx','intersects', 'NEAREST_PT','DISTANCE'])  
    
    gas_intersection = gas_intersection[gas_intersection['intersects']==True]
    
    gas_intersection = gas_intersection.sort_values("DISTANCE").groupby("L_idx", as_index=False).first()
    gas_intersection['END'] = source_df.iloc[gas_intersection['L_idx'].values,source_df.columns.get_loc('unique_id')].values
    gas_intersection['START'] = combined_target_df.iloc[gas_intersection['R_idx'].values,combined_target_df.columns.get_loc('unique_id')].values
    
    out_df = pd.concat([coal_intersection, oil_intersection, gas_intersection])[['START','END','NEAREST_PT','DISTANCE']]
    
    return out_df

def cities_delauney(df_cities, gdf_ne):
    logger = logging.getLogger('flmile'+'_'+'CITIES-DELAUNEY')
    
    df_cities['coordinates'] = df_cities['geometry'].apply(lambda el: wkt.loads(el).representative_point().coords[:][0])
    
    #cities['geometry'] = cities['coordinates'].apply(geometry.Point)
    
    #data = dict(zip(range(len(cities)),cities[['CityNodeId:ID(CityNode)','geometry']].values.tolist()))
    
    logger.info('Getting Delauney triangles')
    delTri = scipy.spatial.Delaunay(df_cities['coordinates'].values.tolist()) # just keep in WGS84, whatever.
    
    edges = set()
    # for each Delaunay triangle
    for n in range(delTri.nsimplex):
        # for each edge of the triangle
        # sort the vertices
        # (sorting avoids duplicated edges being added to the set)
        # and add to the edges set
        edge = sorted([delTri.vertices[n,0], delTri.vertices[n,1]])
        edges.add((edge[0], edge[1]))
        edge = sorted([delTri.vertices[n,0], delTri.vertices[n,2]])
        edges.add((edge[0], edge[1]))
        edge = sorted([delTri.vertices[n,1], delTri.vertices[n,2]])
        edges.add((edge[0], edge[1]))
        
        
    logger.info('Make new edges')
    
    ls_records = []
    for e in tqdm(list(edges)):
        ls_records.append({
            'START':df_cities.iloc[e[0],df_cities.columns.get_loc('unique_id')],
            'END':df_cities.iloc[e[1],df_cities.columns.get_loc('unique_id')],
            'points':[pt for pt in ops.nearest_points(wkt.loads(df_cities.iloc[e[0],df_cities.columns.get_loc('geometry')]), wkt.loads(df_cities.iloc[e[1],df_cities.columns.get_loc('geometry')]))]
    })

    df_lss = pd.DataFrame.from_records(ls_records)
    
    logger.info(f'Getting distance and linestring')
    df_lss['DISTANCE'] = df_lss['points'].progress_apply(lambda el: V_inv((el[0].y, el[0].x),(el[1].y, el[1].x))[0]*1000) #m 
    df_lss['geometry'] = df_lss['points'].progress_apply(lambda el: geometry.LineString([(pt.x, pt.y) for pt in el]))
    
    logger.info(f'Getting coastline and intersection')
    coastline_mp = gdf_ne.geometry.boundary.unary_union # paralellise this
    coastline_df = pd.DataFrame(gpd.GeoDataFrame(list(coastline_mp)).rename(columns={0:'geometry'}).set_geometry('geometry'))
    logger.info(f'casting to str')
    coastline_df['geometry'] = coastline_df['geometry'].progress_apply(lambda el: el.wkt)
    df_lss['geom_str']=df_lss['geometry'].progress_apply(lambda el: el.wkt) # L_idx
    df_lss['dummy']='na'
    
    
    coastline_df['unique_id'] = 'COASTLINE_'+coastline_df.index.astype(str)
    df_lss['unique_id'] = 'CITYFINALMILE_'+df_lss.index.astype(str)
    
    intersection_df = pd.DataFrame(
                        mp_sjoin_mindist(
                            df_linear=coastline_df, 
                            df_buffered=df_lss, 
                            left_geom_column='geometry', 
                            right_geom_column='dummy', 
                            right_buffer_column='geom_str',
                            N_workers=N_WORKERS, 
                            logger=logger,
                            include_min_dist=False),
                        columns=['L_idx','R_idx','intersects','NEAREST_PT', 'DISTANCE'])
    
    intersection_df = intersection_df[intersection_df['intersects']==True]

    
    df_lss = df_lss[~df_lss.index.isin(intersection_df['L_idx'].unique())]
    
    logger.info(f'Got {len(df_lss)} finalmile city-city connections')
    
    return df_lss[['START','END','DISTANCE']]