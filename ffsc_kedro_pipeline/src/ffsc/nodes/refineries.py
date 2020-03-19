import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from typing import Dict, SupportsInt, AnyStr, Tuple
from src.ffsc.nodes.utils import flatten_list_of_lists

from src.ffsc.nodes.pipeline_nodes import coord_to_pipe_key

from src.ffsc.nodes.utils import (
    preprocess_geodata,
    create_nodes,
    create_edges_for_network_connections,
)


def preprocess_refineries_data(data: Dict) -> pd.DataFrame:
    return preprocess_geodata(data)


def _linestring_to_list_of_points(line):
    return list(MultiPoint(line.coords))


def refinery_item_to_node_id(id: SupportsInt) -> AnyStr:
    return "refinery_" + str(int(id))


def create_refinery_graph_components(
    prm_refineries_data: pd.DataFrame,
    prm_refineries_matched_with_pipelines: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols_to_fill = ["Commodity"]
    nodes = create_nodes(
        prm_refineries_data,
        cols_to_fill,
        refinery_item_to_node_id,
        "Refinery",
        "RefineryID",
    )

    edges = create_edges_for_network_connections(
        prm_refineries_matched_with_pipelines,
        "Refinery",
        "RefineryID",
        "PipelineNode",
        "PipelineNodeID",
        refinery_item_to_node_id,
        coord_to_pipe_key,
        "REFINERY_PIPELINE_CONNECTOR",
    )

    return nodes, edges
