import pandas as pd
from typing import Dict, SupportsInt, AnyStr, Tuple
from ffsc.pipeline.nodes.utils import (
    preprocess_geodata,
    create_nodes,
    create_edges_for_network_connections,
)
from ffsc.pipeline.nodes.railways import coord_to_rail_key


def preprocess_coal_mine_data(data: Dict) -> pd.DataFrame:
    df = preprocess_geodata(data)
    return df.rename({"type_x": "mine_type", "type_y": "geometry_type"}, axis=1)


def coal_mine_item_to_node_id(id: SupportsInt) -> AnyStr:
    return "coal_mine_" + str(int(id))


def create_coal_mine_graph_components(
    prm_coal_mine_data: pd.DataFrame, prm_coal_mines_merged_with_railways: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    cols_to_fill = ["md_country", "md_source", "facility_n", "commodity"]
    nodes = create_nodes(
        prm_coal_mine_data,
        cols_to_fill,
        coal_mine_item_to_node_id,
        "CoalMine",
        "CoalMineID",
    )

    edges = create_edges_for_network_connections(
        prm_coal_mines_merged_with_railways,
        "CoalMine",
        "CoalMineID",
        "RailwayNode",
        "RailwayNodeID",
        coal_mine_item_to_node_id,
        coord_to_rail_key,
        "COAL_MINE_RAILAWAY_CONNECTOR",
    )

    return nodes, edges
