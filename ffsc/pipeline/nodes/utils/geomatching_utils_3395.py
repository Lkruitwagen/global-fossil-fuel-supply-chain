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
    
    facility_data = gpd.GeoDataFrame(facility_data, geometry=facility_data['geometry'], crs='epsg:4326')
    facility_data.geometry = facility_data.geometry.to_crs('epsg:3395')
    facility_data['3395_x'] = facility_data.geometry.x
    facility_data['3395_y'] = facility_data.geometry.y
    
    logger.info(f't_f - {time.time()-tic:.1f} - Coordinate manipulation')
    

    #logger.info(f't_s - {time.time()-tic:.1f} - Point buffering on Dask')
    
    #ddf = dd.from_pandas(pipeline_df, npartitions=NPARTITIONS) # bring this from parameters in the future

    #meta= pd.DataFrame({'pipeline_id': [1], 'pipeline_segment_id': [2], 'snapped_geometry':['str']})
    #meta=pd.Series({'geometry':['str']})
    
    #def df_buffer(df,distance):
    #    return df['geometry'].apply(lambda x: geodesic_point_buffer(x.y, x.x, distance))
    
    #facility_data['geometry'] = client.compute(dd.from_pandas(facility_data,npartitions=NPARTITIONS).map_partitions(
    #    df_buffer,
    #    parameters['max_geo_matching_distance'],
    #    meta=meta)).result()#.compute()
    #client.restart()
    
    
    #logger.info(f't_f - {time.time()-tic:.1f} - Point buffering on Dask')
    
    logger.info(f't_s - {time.time()-tic:.1f} - Segment Reduction')

    unique_nodes = unique_nodes_from_segments(network_data.snapped_geometry)
    #meta = pd.Series({'snapped_geometry':[list]})
    #def dfLStocoordlist(df):
    #    return df.snapped_geometry.apply(lambda x: list(x.coords))
    
    #unique_nodes = client.compute(dd.from_pandas(network_data, npartitions=NPARTITIONS).map_partitions(
    #    dfLStocoordlist,
    #    meta=meta)).result()
    #unique_nodes = unique_nodes.explode().unique().tolist()
    #logger.info(f'len unique_nodes {len(unique_nodes)}, {time.time()-tic:.1f}')
    
    logger.info(f't_f - {time.time()-tic:.1f} - Segment Reduction')
    
    logger.info(f't_s - {time.time()-tic:.1f} - Preparing Join')
    
    

    node_df = pd.DataFrame()
    node_df["coordinates"] = unique_nodes
    node_df['geometry'] = node_df['coordinates'].apply(Point)
    logger.info(f't_s - {time.time()-tic:.1f} - Mapping Geometries')
    
    #meta = pd.Series({'geometry':['str']})
    #node_df['geometry'] = client.compute(dd.from_pandas(node_df, npartitions=NPARTITIONS).apply(lambda row: Point(row['coordinates']), meta=meta, axis=1)).result()#.compute()
    #node_df['geometry'] = node_df['coordinates'].apply(Point)
    
    #geo_refineries = gpd.GeoDataFrame(facility_data)
    
    meta = node_df.iloc[0:2,:]
    meta['3995_x'] = 1.
    meta['3995_y'] = 1.
    
    def convert_nodes(df):
        gdf = gpd.GeoDataFrame(df, geometry=df['geometry'], crs='epsg:4326')
        gdf = gdf.to_crs('epsg:3395')
        gdf['3395_x'] = gdf.geometry.x
        gdf['3395_y'] = gdf.geometry.y
        return pd.DataFrame(gdf)
    
    
    # https://stackoverflow.com/questions/41471248/how-to-efficiently-submit-tasks-with-large-arguments-in-dask-distributed
    
    node_df = client.compute(dd.from_pandas(node_df, npartitions=NPARTITIONS).map_partitions(convert_nodes, meta=meta)).result()#.compute()
    
    logger.info(f't_f - {time.time()-tic:.1f} - Mapped geometries')
    logger.info(f't_f - {time.time()-tic:.1f} - buffering intersection')
    
    [ndf_fut] = client.scatter([node_df])
    
    def buffered_nodes(row, ndf_fut):
        indices = ((ndf_fut['3395_x']-row['3395_x'])**2 + (ndf_fut['3395_y']-row['3395_y'])**2)**(1/2) <= parameters['max_geo_matching_distance']
        return ndf_fut.loc[indices,'coordinates'].values.tolist()
    
    facility_data['network_coordinates'] = client.compute(
        dd.from_pandas(
            facility_data, npartitions=NPARTITIONS) \
          .apply(buffered_nodes,args=(ndf_fut,), axis=1, meta=pd.Series({'network_coordinates':list}))).result()
                                                          
    matched = facility_data[facility_data['network_coordinates'].str.len()>0]
    matched = matched.explode('network_coordinates').rename(columns={'coordinates':'facility_coordinates'})
    
    
    #def dask_sjoin(df, network_gdf):
    #    # df = facility_df
    #    # df['geometry'] = df["coordinates"].apply(Point)
    #    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'], crs='epsg:4326')
    #    gdf = gpd.sjoin(gdf, network_gdf, how='inner',op='contains')
    #
    #   return pd.DataFrame(gdf)
    
    #meta = facility_data.iloc[0:2,:]
    #meta['coordinates_right']=list
    #meta = meta.rename(columns={'coordinates':'coordinates_left'})
    
    #logger.info(f'spatial join on dask {time.time()-tic}')
    #matched = client.compute(dd.from_pandas(facility_data, npartitions=NPARTITIONS).map_partitions(
    #    dask_sjoin,
    #    geo_pipelines,
    #    meta=meta)).result()#.compute()
    #client.restart()
    
    #logger.info(f'spatial join on dask {time.time()-tic}')
    
    #matched = gpd.sjoin(geo_refineries, geo_pipelines, op="contains", how="inner")

    #matched = pd.DataFrame(matched)

    #matched = matched.rename(
    #    {
    #        "coordinates_left": "facility_coordinates",
    #        "coordinates_right": "network_coordinates",
    #    },
    #    axis=1,
    #)
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

