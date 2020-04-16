import logging, sys, time

from src.ffsc.nodes.intersections import find_intersecting_points
from src.ffsc.nodes.utils import (
    preprocess_geodata,
    unique_nodes_from_segments,
    convert_segments_to_lines,
    calculate_havesine_distance,
)

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
from typing import List, AnyStr

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

NPARTITIONS=6
#from dask.distributed import Client
#client = Client(processes=4)
import dask.dataframe as dd



def preprocess_railway_data_int(data):
    df = preprocess_geodata(data, "railway_id")
    df = df.rename({"type_x": "rail_type", "type_y": "geometry_type"}, axis=1)
    # The country for all rows with missing region was either one or several of following:
    # United States of America, Canada, and Mexico
    # or missing.
    df.loc[df.md_country.notnull(), "md_region"] = df.loc[
        df.md_country.notnull(), "md_region"
    ].fillna("N. and C. America")
    return df[
        [
            "railway_id",
            "md_country",
            "md_region",
            "rail_type",
            "geometry_type",
            "coordinates",
        ]
    ]


def preprocess_railway_data_prm(int_railways, parameters):
    logger = logging.getLogger(__name__)
    tic = time.time()
    
    def _nest_coords(row):
        if row['geometry_type']=='MultiLineString':
            return row['coordinates']
        else:
            return [list(row['coordinates'])]
        
    # nest any LineStrings to [[[],[]]]
    logger.info(f'nesting coordinates {time.time()-tic}')
    int_railways['coordinates'] = int_railways.apply(lambda row: _nest_coords(row), axis=1)
    logger.info(f'{int_railways["coordinates"].str.len().unique()}')
    logger.info(f'nesting coordinates {time.time()-tic}')
    
    
    #explode along coordinates
    logger.info(f'exploding df {time.time()-tic}')
    railway_df = int_railways.explode('coordinates')
    railway_df['railway_segment_id'] = railway_df.index
    logger.info(f'{railway_df["coordinates"].str.len().unique()}')
    logger.info(f'exploded df {time.time()-tic}')
    
    
    # get the missing region
    logger.info(f'get missing regions {time.time()-tic}')
    railway_missing_region_df = railway_df.loc[railway_df.md_region.isna()].copy()
    logger.info(f'get missing regions {time.time()-tic}')
    
    # cast to dask dataframe
    logger.info(f'cast to dask {time.time()-tic}')
    ddf = dd.from_pandas(railway_missing_region_df, npartitions=NPARTITIONS)
    logger.info(f'cast to dask {time.time()-tic}')
    
    # load NE
    logger.info(f'get world gdf {time.time()-tic}')
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world.geometry = world.geometry.geometry.buffer(10)
    logger.info(f'get world gdf {time.time()-tic}')
    
    ### parallelise sjoin in Dask
    
    # define meta dataframe
    logger.info(f'get meta {time.time()-tic}')
    meta = pd.DataFrame(columns=['railway_id','name','continent'])
    meta.railway_id = meta.railway_id.astype(int)
    meta.name = meta.name.astype('str')
    meta.continent = meta.continent.astype('str')
    logger.info(f'get meta {time.time()-tic}')
    
    def dask_sjoin(df, obj_col):

        df[obj_col] = df["coordinates"].apply(geometry.LineString)
        gdf = gpd.GeoDataFrame(df, geometry=df[obj_col], crs={'init':'epsg:4326'})
        gdf = gpd.sjoin(gdf, world[['name','continent','geometry']], how='left',op='intersects')

        return pd.DataFrame(gdf[['railway_id','name','continent']])
    
    logger.info(f'map regions on dask {time.time()-tic}')
    retrieved_region_df = ddf.map_partitions(dask_sjoin,'railway_object', meta=meta).compute()
    logger.info(f'map regions on dask {time.time()-tic}')
    #client.restart()
    
    logger.info(f'regions reset {time.time()-tic}')
    region_dict = {
        "Oceania": "Australia and Oceania",
        "Africa": "Africa",
        "North America": "N. and C. America",
        "Asia": "Asia",
        "South America": "South America",
        "Europe": "Europe",
    }

    retrieved_region_df["retrieved_region"] = retrieved_region_df.continent.map(
        region_dict
    )

    retrieved_region_df.loc[
        retrieved_region_df.retrieved_region == "Asia", "retrieved_region"
    ] = retrieved_region_df.loc[
        retrieved_region_df.retrieved_region == "Asia", "name"
    ].apply(
        lambda x: "Middle East"
        if x
        in [
            "Iraq",
            "Turkey",
            "Armenia",
            "Azerbaijan",
            "Iran",
            "Kuwait",
            "Israel",
            "Jordan",
            "Syria",
            "Saudi Arabia",
            "Lebanon",
        ]
        else "Asia"
    )

    retrieved_region_df.dropna(subset=["retrieved_region"], inplace=True)

    retrieved_region_df = (
        retrieved_region_df.drop_duplicates(subset=["railway_id", "retrieved_region"])
        .sort_values(["railway_id", "retrieved_region"])
        .groupby("railway_id")
        .retrieved_region.apply(lambda x: "; ".join(x.tolist()))
        .reset_index()
    )
    logger.info(f'regions reset {time.time()-tic}')

    
    logger.info(f'merge back to main df {time.time()-tic}')
    railway_df = railway_df.merge(retrieved_region_df, how="left", on="railway_id")

    railway_df.loc[:, "md_region"] = railway_df.loc[:, "md_region"].fillna(
        railway_df.loc[:, "retrieved_region"]
    )

    railway_df.drop(columns=["retrieved_region"], inplace=True)
    logger.info(f'merge back to main df {time.time()-tic}')
    
    # drop the regionless orphans
    railway_df.dropna(subset=['md_region'], inplace=True)

    logger.info(f'making geometry column {time.time()-tic}')
    railway_df['railway_object'] = railway_df.coordinates.apply(geometry.LineString)
    logger.info(f'making geometry column {time.time()-tic}')

    logger.info(f'making new ddf {time.time()-tic}')
    ddf = dd.from_pandas(railway_df, npartitions=NPARTITIONS) # bring this from parameters in the future

    logger.info(f'making new ddf {time.time()-tic}')
    meta= pd.DataFrame({'railway_id': [1], 'railway_segment_id': [2], 'snapped_geometry':['str']})
    


    logger.info(f'calling groupby md_region{time.time()-tic}')
    prm_railways_data = ddf.groupby(['md_region']) \
                            .apply(find_intersecting_points, 
                                    parameters, 
                                    'railway_object',
                                    ['railway_id','railway_segment_id'], 
                                    meta=meta) \
                            .compute()

    prm_railways_data = prm_railways_data[
        ["railway_segment_id", "railway_id", "snapped_geometry"]
    ].copy()

    # Amend the pipelines that did not have any intersections.
    prm_railways_data = pd.concat(
        [
            railway_df.loc[
                ~railway_df.railway_segment_id.isin(
                    prm_railways_data.railway_segment_id.unique()
                ),
                ["railway_segment_id", "railway_id", "railway_object"],
            ].rename(columns={"railway_object": "snapped_geometry"}),
            prm_railways_data,
        ],
        ignore_index=True,
    ).reset_index(drop=True)

    return prm_railways_data


def coord_to_rail_key(coord: List) -> AnyStr:
    return "railway_node_" + "".join([str(item) for item in coord])


def create_railway_graph_components(prm_railways_data):
    unique_nodes = unique_nodes_from_segments(prm_railways_data.snapped_geometry)

    node_df = pd.DataFrame()
    node_df["coordinates"] = unique_nodes
    node_df["RailwayNodeID:ID(RailwayNode)"] = [
        coord_to_rail_key(coord) for coord in unique_nodes
    ]
    node_df["lat"] = [coord[0] for coord in unique_nodes]
    node_df["long"] = [coord[1] for coord in unique_nodes]
    node_df[":LABEL"] = "RailwayNode"

    edges = convert_segments_to_lines(
        [list(line.coords) for line in prm_railways_data.snapped_geometry]
    )

    edge_df = pd.DataFrame()
    edge_df["StartNodeId:START_ID(RailwayNode)"] = [
        coord_to_rail_key(edge[0]) for edge in edges
    ]
    edge_df["EndNodeId:END_ID(RailwayNode)"] = [
        coord_to_rail_key(edge[1]) for edge in edges
    ]
    edge_df[":TYPE"] = "RAILWAY_CONNECTION"

    edge_df["distance"] = calculate_havesine_distance(
        np.array([edge[0] for edge in edges]), np.array([edge[1] for edge in edges])
    )
    edge_df["impedance"] = edge_df["distance"] ** 2

    return node_df, edge_df
