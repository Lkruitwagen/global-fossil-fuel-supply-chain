import pandas as pd
import numpy as np
from typing import AnyStr, List, Callable


def create_id_col_name(
    node_id_col_name: AnyStr, node_label: AnyStr, id_val: AnyStr = "ID"
) -> AnyStr:
    return node_id_col_name + f":{id_val}({node_label})"


def create_nodes(
    node_data: pd.DataFrame,
    cols_to_fill: List[AnyStr],
    node_id_transformation_func: Callable,
    node_label: AnyStr,
    node_id_col_name: AnyStr,
) -> pd.DataFrame:
    """
    Reformats a table to meet Neo4js node table requirements
    :param node_data: The table to be reformatted
    :param cols_to_fill: Columns that will be filled with "unknown".
    All other columns containing missing values will be dropped.
    :param node_id_transformation_func: Function to turn a numeric node id into a string
    :param node_label: Neo4j node label to be applied
    :param node_id_col_name: Name to give id column
    :return: Dataframe formatted for use with the neo4j-admin import tool
    """

    node_identifier = create_id_col_name(node_id_col_name, node_label)
    nodes = node_data.rename({"item_id": node_identifier}, axis=1)

    nodes[cols_to_fill] = nodes[cols_to_fill].fillna("Unknown")
    for col in cols_to_fill:
        nodes[col] = nodes[col].str.replace("\W", "")

    nodes = nodes.dropna(axis=1)

    nodes[node_identifier] = nodes[node_identifier].apply(node_id_transformation_func)

    nodes[":LABEL"] = node_label

    return nodes


def calculate_euclidean_distance(
    coord_a: np.ndarray, coord_b: np.ndarray
) -> np.ndarray:
    """Computes the pairwise euclidean distance between 2 2d arrays"""
    return np.linalg.norm(coord_a - coord_b, axis=1)


def calculate_havesine_distance(coord_a: np.ndarray, coord_b: np.ndarray) -> np.ndarray:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lat1 = coord_a[:, 0]
    lon1 = coord_a[:, 1]
    lat2 = coord_b[:, 0]
    lon2 = coord_b[:, 1]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def create_edges_for_network_connections(
    edge_data: pd.DataFrame,
    start_node_label: AnyStr,
    start_node_id_col_name: AnyStr,
    end_node_label: AnyStr,
    end_node_id_col_name: AnyStr,
    start_node_id_transformation_func: Callable,
    end_node_id_transformation_func: Callable,
    edge_type: AnyStr,
) -> pd.DataFrame:
    """
    Creates a dataframe containing edges (relationships) meeting Neo4j edge table requirements
    :param edge_data: Table to be formatted
    :param start_node_label: Neo4j node label to be applied to the start node
    :param start_node_id_col_name: Name to give start node id column
    :param end_node_label: Neo4j node label to be applied to the end node
    :param end_node_id_col_name: Name to give end node id column
    :param start_node_id_transformation_func: Function to turn the numeric node id of the start node into a string
    :param end_node_id_transformation_func: Function to turn the numeric node id of the start node into a string
    :param edge_type: Neo4j type of edge
    :return:
    """

    start_node_identifier = create_id_col_name(
        start_node_id_col_name, start_node_label, "START_ID"
    )

    end_node_identifier = create_id_col_name(
        end_node_id_col_name, end_node_label, "END_ID"
    )

    edges = edge_data[["item_id", "facility_coordinates", "network_coordinates"]]
    edges["distance"] = calculate_havesine_distance(
        np.array([*edges.facility_coordinates]), np.array([*edges.network_coordinates])
    )
    edges["impedance"] = edges.distance ** 2

    edges.loc[:, start_node_identifier] = edges["item_id"].apply(
        start_node_id_transformation_func
    )

    edges.loc[:, end_node_identifier] = edges["network_coordinates"].apply(
        end_node_id_transformation_func
    )

    edges = edges.drop(
        ["item_id", "facility_coordinates", "network_coordinates"], axis=1
    )

    edges[":TYPE"] = edge_type

    return edges
