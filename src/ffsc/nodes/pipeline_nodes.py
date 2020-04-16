from src.ffsc.nodes.intersections import find_intersecting_points
from src.ffsc.nodes.utils.geo_json_utils import preprocess_geodata
from src.ffsc.nodes.utils import calculate_havesine_distance
from shapely import geometry, ops
import pandas as pd
import numpy as np
import geopandas as gpd
import itertools, logging, sys, time
from functools import partial
from typing import List, Dict, AnyStr


from shapely.geometry import Point, MultiPoint

NPARTITIONS=6
#from dask.distributed import Client
#client = Client(processes=4)
import dask.dataframe as dd


from src.ffsc.nodes.utils import unique_nodes_from_segments, convert_segments_to_lines


def preprocess_pipeline_data_int(raw_pipelines_data):
    int_pipelines_data = preprocess_geodata(raw_pipelines_data, "pipeline_id")
    # The country for all rows with missing region was one or several of following:
    # United States of America, Canada, and Mexico
    int_pipelines_data.md_region.fillna("N. and C. America", inplace=True)

    return int_pipelines_data


def preprocess_pipeline_data_prm(int_pipelines_data, parameters):

    """
    This function breaks down the pipeline objects into single LineString objects, then for each region calls a function
    that finds the intersection of pipelines and snap the intersecting points into the pipeline LineString objects.
    :int_pipelines_data: int_pipelines_data: intermediate pipeline data
    :parameters: pipeline parameters
    :return: dataframe with pipeline segment id, pipeline id, and snapped LineString objects
    """

    logger = logging.getLogger(__name__)
    tic = time.time()
    
    def _nest_coords(row):
        if row['geometry_type']=='MultiLineString':
            return row['coordinates']
        else:
            return [list(row['coordinates'])]

    # Differentiate between the column names
    int_pipelines_data.rename(
        columns={"type_x": "pipeline_type", "type_y": "geometry_type"}, inplace=True
    )

    # Break down the pipelines with multiple LineString objects in their geometry. Preserve the original pipeline_id,
    # but also create a unqiue pipeline_segment_id

        
    # nest any LineStrings to [[[],[]]]
    logger.info(f'nesting coordinates {time.time()-tic}')
    int_pipelines_data['coordinates'] = int_pipelines_data.apply(lambda row: _nest_coords(row), axis=1)
    logger.info(f'{int_pipelines_data["coordinates"].str.len().unique()}')
    logger.info(f'nesting coordinates {time.time()-tic}')
    
    
    #explode along coordinates
    logger.info(f'exploding df {time.time()-tic}')
    pipeline_df = int_pipelines_data.explode('coordinates')
    pipeline_df['pipeline_segment_id'] = pipeline_df.index
    logger.info(f'{pipeline_df["coordinates"].str.len().unique()}')
    logger.info(f'exploded df {time.time()-tic}')



    # Convert to shapely object
    logger.info(f'converging to shapely {time.time()-tic}')
    pipeline_df["pipeline_object"] = pipeline_df["coordinates"].apply(
        geometry.LineString
    )
    logger.info(f'converting to shapely {time.time()-tic}')

    #merging regions not necessary because we exploded
    #logger.info(f'merging regions {time.time()-tic}')
    #logger.info(f'pipeline df columns {pipeline_df.columns}')

    # Bring in region information for each pipeline (we use region to spead up the process of finding intersections)
    #pipeline_df = pipeline_df.merge(
    #    int_pipelines_data[["pipeline_id", "md_region"]], on="pipeline_id"
    #)
    #logger.info(f'merging regions {time.time()-tic}')

    logger.info(f'doing dask')
    ddf = dd.from_pandas(pipeline_df, npartitions=NPARTITIONS) # bring this from parameters in the future

    meta= pd.DataFrame({'pipeline_id': [1], 'pipeline_segment_id': [2], 'snapped_geometry':['str']})

    # For each region call the function that finds the intersections and adds a corresponding breaking point to the
    # LineString object.
    prm_pipelines_data = ddf.groupby("md_region").apply(find_intersecting_points, 
                                                        parameters=parameters, 
                                                        meta=meta)

    prm_pipelines_data = prm_pipelines_data[
        ["pipeline_segment_id", "pipeline_id", "snapped_geometry"]
    ].copy()

    # Amend the pipelines that did not have any intersections.
    prm_pipelines_data = pd.concat(
        [
            pipeline_df.loc[
                ~pipeline_df.pipeline_segment_id.isin(
                    prm_pipelines_data.pipeline_segment_id.unique()
                ),
                ["pipeline_segment_id", "pipeline_id", "pipeline_object"],
            ].rename(columns={"pipeline_object": "snapped_geometry"}),
            prm_pipelines_data,
        ],
        ignore_index=True,
    ).reset_index(drop=True)

    return prm_pipelines_data


def coord_to_pipe_key(coord: List) -> AnyStr:
    return "pipe_node_" + "".join([str(item) for item in coord])


def create_pipeline_graph_tables(prm_pipeline_data: pd.DataFrame, parameters: Dict):

    unique_nodes = unique_nodes_from_segments(prm_pipeline_data.snapped_geometry)

    node_df = pd.DataFrame()
    node_df["coordinates"] = unique_nodes
    node_df["PipeNodeID:ID(PipelineNode)"] = [
        coord_to_pipe_key(coord) for coord in unique_nodes
    ]
    node_df["lat"] = [coord[0] for coord in unique_nodes]
    node_df["long"] = [coord[1] for coord in unique_nodes]
    node_df[":LABEL"] = "PipelineNode"

    edges = convert_segments_to_lines(
        [list(line.coords) for line in prm_pipeline_data.snapped_geometry]
    )

    edge_df = pd.DataFrame()
    edge_df["StartNodeId:START_ID(PipelineNode)"] = [
        coord_to_pipe_key(edge[0]) for edge in edges
    ]
    edge_df["EndNodeId:END_ID(PipelineNode)"] = [
        coord_to_pipe_key(edge[1]) for edge in edges
    ]
    edge_df[":TYPE"] = "PIPELINE_CONNECTION"

    edge_df["distance"] = calculate_havesine_distance(
        np.array([edge[0] for edge in edges]), np.array([edge[1] for edge in edges])
    )
    edge_df["impedance"] = edge_df["distance"] ** 2

    return node_df, edge_df
