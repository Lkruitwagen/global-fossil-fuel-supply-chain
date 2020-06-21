#from ffsc.pipeline.nodes.intersections import find_intersecting_points
from ffsc.pipeline.nodes.utils.geo_json_utils import preprocess_geodata
from ffsc.pipeline.nodes.utils import calculate_havesine_distance
from shapely import geometry, ops
import pandas as pd
import numpy as np
import geopandas as gpd
gpd.options.use_pygeos=False
import itertools, logging, sys, time
from functools import partial
from typing import List, Dict, AnyStr


from shapely.geometry import Point, MultiPoint
from shapely import ops

import multiprocessing as mp
NPARTITIONS=mp.cpu_count()//8




from ffsc.pipeline.nodes.utils import unique_nodes_from_segments, convert_segments_to_lines


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
    
    # lazily create dask cluster
    from dask.distributed import Client
    client = Client(n_workers=NPARTITIONS,memory_limit=((60-4)//NPARTITIONS)*1e9) #GB
    client.cluster
    import dask.dataframe as dd

    logger = logging.getLogger(__name__)
    tic = time.time()
    
    def find_intersecting_points(
        network_df,
        parameters,
        object_column="pipeline_object",
        entity_ids=["pipeline_id", "pipeline_segment_id"],
    ):
        """
        This function finds the intersecting LineString objects, finds the intersecting points, and then snap the
        intersecing points to the LineString objects.

        :param network_df: The dataframe with the LineString objects
        :param parameters: Dictionary with pipeline parameters.
        :param object_column: The column in the network_df containing the LineString objects.
        :param entity_ids: The unique keys of the dataframe.
        :return: Returns the dataframe with the LineString objects modified to include the intersecting points.
        """
        # Set up the GeoPandas dataframe
        network_gdf = gpd.GeoDataFrame(network_df, geometry=network_df[object_column])
        network_gdf.sindex

        # Find the intersecting LineStrings
        intersected_gdf = gpd.sjoin(
            network_gdf[entity_ids + ["geometry"]],
            network_gdf[entity_ids + ["geometry"]],
            op="intersects",
        )

        # Bring in the LineString object of the other intersecting object:
        intersected_gdf = (
            intersected_gdf.merge(
                network_gdf[["geometry"]], left_on="index_right", right_index=True
            )
            .reset_index(drop=True)
            .rename(columns={"geometry_x": "geometry_left", "geometry_y": "geometry_right"})
        )

        # Find the intersection of the intersecting LineStrings:
        if not intersected_gdf.empty:
            intersected_gdf["intersection"] = intersected_gdf.apply(
                lambda row: row["geometry_left"].intersection(row["geometry_right"]), axis=1
            )
        else:
            intersected_gdf["intersection"] = None

        # We only focus on the intersection where the type is either Point or MultiPoint.
        # This removes the intersection of LineStrings with themselves.
        # orig_index is the index for intersection.
        intersected_gdf['geomtype'] = intersected_gdf.intersection.apply(lambda x: x.type)

        
        points_df = intersected_gdf.loc[
                intersected_gdf.intersection.apply(lambda x: "Point" in x.type) # get only the points
            ].intersection.apply(
                lambda x: list(x)
                if x.type == "MultiPoint"
                else [x]
            ).explode().reset_index().rename(columns={'index':'orig_index','intersection':'intersection_point'})
        

        # Remove the duplicated intersections (points appear twice per intersection.)
        points_gdf = gpd.GeoDataFrame(
            points_df["orig_index"], geometry=points_df["intersection_point"]
        ).drop_duplicates()

        # Bring in the entity_ids and geometries of LinesStrings corresponding for each intersection.
        intersection_df = (
            points_gdf.merge(
                intersected_gdf[
                    [
                        entity_id + "_" + direction
                        for direction in ["left", "right"]
                        for entity_id in entity_ids + ["geometry"]
                    ]
                ],
                left_on="orig_index",
                right_index=True,
            )
            .rename(columns={"geometry": "intersection_point"})
            .drop(columns=["orig_index"])
        ).reset_index(drop=True)

        # Put the left and right LineStrings on top of each other.
        intersection_df = (
            (
                pd.concat(
                    [
                        intersection_df[
                            ["intersection_point"]
                            + [
                                entity_id + "_left"
                                for entity_id in ["geometry"] + entity_ids
                            ]
                        ].rename(
                            columns={
                                entity_id + "_left": entity_id
                                for entity_id in ["geometry"] + entity_ids
                            }
                        ),
                        intersection_df[
                            ["intersection_point"]
                            + [
                                entity_id + "_right"
                                for entity_id in ["geometry"] + entity_ids
                            ]
                        ].rename(
                            columns={
                                entity_id + "_right": entity_id
                                for entity_id in ["geometry"] + entity_ids
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            )
            .reset_index(drop=True)
            .drop_duplicates()
        )

        # For each LineString, combine all intersecting points to a single MultiPoint object.

        intersecting_points = (
            intersection_df.groupby(entity_ids)
            .intersection_point.apply(lambda x: gpd.GeoDataFrame(geometry=x).unary_union)
            .reset_index()
        )

        for entity_id in entity_ids:
            if entity_id not in intersecting_points.columns:
                intersecting_points[entity_id]=None
        
        # Bring in entity ids and geometries of intersecting LineStrings to the objects above.
        intersecting_points = intersecting_points.merge(
            intersection_df[entity_ids + ["geometry"]].drop_duplicates(), on=entity_ids
        )


        if not intersecting_points.empty:
            
            # Snap the intersecting points to LineString objects.
            intersecting_points["snapped_geometry"] = intersecting_points.apply(
                lambda row: ops.snap(
                    row["geometry"], row["intersection_point"], parameters["snapping_threshold"]
                ),
                axis=1,
            )
        else:
            intersecting_points["snapped_geometry"] = None

        return intersecting_points[entity_ids + ['snapped_geometry']]
    
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
    logger.info(f'nesting coordinates {time.time()-tic}')
    
    
    #explode along coordinates
    logger.info(f'exploding df {time.time()-tic}')
    pipeline_df = int_pipelines_data.explode('coordinates')
    pipeline_df['pipeline_segment_id'] = pipeline_df.index
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
    print (pipeline_df.groupby('md_region').size())
    ddf = client.persist(dd.from_pandas(pipeline_df, npartitions=len(pipeline_df.md_region.unique())).set_index('md_region')) # pipelines bigger than rail, use fewer partitions


    meta= pd.DataFrame({'pipeline_id': [1], 'pipeline_segment_id': [2], 'snapped_geometry':['str']})

    # For each region call the function that finds the intersections and adds a corresponding breaking point to the
    # LineString object.
    prm_pipelines_data = client.compute(ddf.map_partitions(find_intersecting_points, 
                                                        parameters=parameters, 
                                                        meta=meta)).result()#.compute()

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
