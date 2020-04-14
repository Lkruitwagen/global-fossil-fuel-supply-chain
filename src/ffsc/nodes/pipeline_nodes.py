from src.ffsc.nodes.intersections import find_intersecting_points
from src.ffsc.nodes.utils.geo_json_utils import preprocess_geodata
from src.ffsc.nodes.utils import calculate_havesine_distance
from shapely import geometry, ops
import pandas as pd
import numpy as np
import geopandas as gpd
import itertools
from functools import partial
from typing import List, Dict, AnyStr


from shapely.geometry import Point, MultiPoint


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
    # Differentiate between the column names
    int_pipelines_data.rename(
        columns={"type_x": "pipeline_type", "type_y": "geometry_type"}, inplace=True
    )

    # Break down the pipelines with multiple LineString objects in their geometry. Preserve the original pipeline_id,
    # but also create a unqiue pipeline_segment_id
    pipeline_df = (
        int_pipelines_data[["pipeline_id", "geometry_type", "coordinates"]]
        .apply(
            lambda row: pd.Series(row["coordinates"])
            if row["geometry_type"] == "MultiLineString"
            else pd.Series([list(row["coordinates"])]),
            axis=1,
        )
        .stack()
        .reset_index()
        .drop(columns=["level_1"])
        .rename(columns={"level_0": "pipeline_id", 0: "coordinates"})
        .reset_index()
        .rename(columns={"index": "pipeline_segment_id"})
    )
    # Convert to shapely object
    pipeline_df["pipeline_object"] = pipeline_df["coordinates"].apply(
        geometry.LineString
    )
    # Bring in region information for each pipeline (we use region to spead up the process of finding intersections)
    pipeline_df = pipeline_df.merge(
        int_pipelines_data[["pipeline_id", "md_region"]], on="pipeline_id"
    )

    # For each region call the function that finds the intersections and adds a corresponding breaking point to the
    # LineString object.
    prm_pipelines_data = (
        pipeline_df.groupby("md_region")
        .apply(find_intersecting_points, parameters=parameters)
        .reset_index()
    )

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
