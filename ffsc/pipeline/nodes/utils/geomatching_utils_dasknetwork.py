import logging, time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import transform
from typing import Dict
from functools import partial
import pyproj
from .python_helpers import unique_nodes_from_segments

gpd.options.use_pygeos=False


import multiprocessing as mp
NPARTITIONS=mp.cpu_count()//4

from dask.distributed import Client
client = Client(n_workers=NPARTITIONS,memory_limit=((124-4)//NPARTITIONS)*1e9) #GB
client.cluster
import dask.dataframe as dd
#from dask.multiprocessing import get

proj_wgs84 = pyproj.Proj(init='epsg:4326')

def merge_facility_with_transportation_network_graph(
    network_data: pd.DataFrame, facility_data: pd.DataFrame, parameters: Dict
) -> pd.DataFrame:
    """
    Merges facility data (e.g. refineries) with network data (pipelines or railways).
    A maximum distance given in the parameters is used. This means, one facility can be linked to many network nodes
    and one network node can be linked to many facilities.
    :param network_data: A dataframe containing line segments that form a network.
    We assume that the column containing the line segments is called "snapped_geometry"
    :param facility_data: A dataframe containing facilities as points.
    We assume that the facility location is present as a list of coordinates ina  column called coordinates
    :param parameters: Parameters from the kedro parameters.yml file. We assume it contains a max_geo_matching_distance
    :return: Dataframe facilities merged with network data.
    Retains all attributes of the facilities but only the location of network nodes.
    """
    
    def geodesic_point_buffer(lat, lon, km):
        # Azimuthal equidistant projection
        aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
        project = partial(
            pyproj.transform,
            pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
            proj_wgs84)
        buf = Point(0, 0).buffer(km * 1000)  # distance in metres
        return Polygon(transform(project, buf).exterior.coords[:])
    
    logger = logging.getLogger(__name__)
    tic = time.time()
    
    logger.info(f't_s - {time.time()-tic:.1f} - Merging facility')
    logger.info(f't_s - {time.time()-tic:.1f} - Coordinate manipulation')

    facility_data["coord_x_"] = facility_data.coordinates.apply(lambda x: x[0])
    facility_data["coord_y_"] = facility_data.coordinates.apply(lambda x: x[1])
    facility_data = facility_data.loc[(facility_data.coord_x_ <= 180) &
                                      (facility_data.coord_x_ >= -180) &
                                      (facility_data.coord_y_ <= 90) &
                                      (facility_data.coord_y_ >= -90)]

    facility_data.drop(columns=["coord_x_", "coord_y_"], inplace=True)

    facility_data["geometry"] = facility_data.coordinates.apply(Point)
    
    logger.info(f't_f - {time.time()-tic:.1f} - Coordinate manipulation')
    

    logger.info(f't_s - {time.time()-tic:.1f} - Point buffering on Dask')
    
    #ddf = dd.from_pandas(pipeline_df, npartitions=NPARTITIONS) # bring this from parameters in the future

    #meta= pd.DataFrame({'pipeline_id': [1], 'pipeline_segment_id': [2], 'snapped_geometry':['str']})
    meta=pd.Series({'geometry':['str']})
    
    def df_buffer(df,distance):
        return df['geometry'].apply(lambda x: geodesic_point_buffer(x.y, x.x, distance))
    
    facility_data['geometry'] = client.compute(dd.from_pandas(facility_data,npartitions=NPARTITIONS*4).map_partitions(
        df_buffer,
        parameters['max_geo_matching_distance'],
        meta=meta)).result()#.compute()

    
    ### join the other way - df on to network segment, not network onto df segment
    logger.info(f't_f - {time.time()-tic:.1f} - Point buffering on Dask')
    
    logger.info(f't_s - {time.time()-tic:.1f} - Segment Reduction')
    
    meta = facility_data.iloc[0:2,:]
    meta['coordinates_right']=list
    meta['index_right']='str'
    meta = meta.rename(columns={'coordinates':'coordinates_left'})
    meta.columns = sorted(meta.columns)
    
    #[df_fut] = client.scatter([facility_data], broadcast=True)
    
    
    def join_df_to_seg(seg_df, df_fut):
        
        # get unique nodes in seg 
        node_df = seg_df.snapped_geometry.apply(lambda x: list(x.coords))
        
        node_df = pd.DataFrame(node_df.explode()).drop_duplicates().rename(columns={'snapped_geometry':'coordinates'})
        #unique_nodes = None # release this memory
        #print ('node_df')
        print ('len node_df',len(node_df))
        
        # make them into Point geoms
        node_df["geometry"] = node_df['coordinates'].apply(Point)
                
        # join them onto df
        node_df = gpd.GeoDataFrame(node_df, geometry=node_df['geometry'], crs='epsg:4326')
        gdf = gpd.GeoDataFrame(df_fut, geometry=df_fut['geometry'], crs='epsg:4326')
        gdf = gpd.sjoin(gdf, node_df, how='inner',op='contains')
        del node_df
        del df_fut
        
        gdf = pd.DataFrame(gdf)
        gdf.columns = sorted(gdf.columns)
        
        return gdf

    logger.info(f't_f - {time.time()-tic:.1f} - Segment Reduction')
    #matched = client.compute(dd.from_pandas(network_data, npartitions=NPARTITIONS*4).map_partitions(
    #    join_df_to_seg,
    #    df_fut,
    #    meta=meta)).result()#.compute()

    
    matched = dd.from_pandas(network_data, npartitions=NPARTITIONS*4).map_partitions(
        join_df_to_seg,
        facility_data,
        meta=meta).compute()
        
    # remove any duplicates
    logger.info(f't_s - {time.time()-tic:.1f} - Removing Duplicates')
    matched = matched.drop_duplicates(subset=['coordinates_left','coordinates_right'])
    logger.info(f't_f - {time.time()-tic:.1f} - Removing Duplicates')
    

    matched = matched.rename(
        {
            "coordinates_left": "facility_coordinates",
            "coordinates_right": "network_coordinates",
        },
        axis=1,
    )
    logger.info(f't_f - {time.time()-tic:.1f} - Done Join')

    return matched

def geodesic_point_buffer(lat, lon, km):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    return Polygon(transform(project, buf).exterior.coords[:])

