import pandas as pd
from .utils import preprocess_geodata
from typing import Tuple


def preprocess_shipping_data(data):

    df = preprocess_geodata(data)

    df["starting_point"] = [coord[0] for coord in df.coordinates]

    df["end_point"] = [coord[1] for coord in df.coordinates]

    return df


def shipping_node_item_to_node_id(id):
    return "shipping_node_" + str(int(id))


def get_unique_shipping_nodes(from_nodes, to_nodes):
    return list(set(from_nodes).union(set(to_nodes)))


def create_shipping_graph_tables(prm_shipping_routes_data: pd.DataFrame):
    prm_shipping_routes_data["From Node0"] = prm_shipping_routes_data[
        "From Node0"
    ].apply(shipping_node_item_to_node_id)
    prm_shipping_routes_data["To Node0"] = prm_shipping_routes_data["To Node0"].apply(
        shipping_node_item_to_node_id
    )

    all_nodes = get_unique_shipping_nodes(
        prm_shipping_routes_data["From Node0"], prm_shipping_routes_data["To Node0"]
    )

    shipping_nodes_df = pd.DataFrame()
    shipping_nodes_df["ShippingNodeID:ID(ShippingNode)"] = all_nodes
    shipping_nodes_df[":LABEL"] = "ShippingNode"

    shipping_edge_df = prm_shipping_routes_data.rename(
        {
            "From Node0": "StartNodeId:START_ID(ShippingNode)",
            "To Node0": "EndNodeId:END_ID(ShippingNode)",
            "id": "RouteId",
            "Impedence0": "impedance",
            "Name0": "RouteName",
            "Length0": "length",
        },
        axis=1,
    ).drop(["item_id", "type"], axis=1)

    shipping_edge_df[":TYPE"] = "SHIPPING_ROUTE"

    return shipping_nodes_df, shipping_edge_df
