import pandas as pd
from shapely import geometry, ops
from joblib import Parallel, delayed

from ffsc.pipeline.nodes.utils import (
    preprocess_geodata,
    create_nodes,
    create_edges_for_network_connections,
)
from ffsc.pipeline.nodes.pipeline_nodes import coord_to_pipe_key
from ffsc.pipeline.nodes.shipping_nodes import shipping_node_item_to_node_id


def preprocess_lng_data(data):
    df = preprocess_geodata(data)
    return df.rename({"type_x": "facility_type", "type_y": "geometry_type"}, axis=1)


def lng_item_to_node_id(id):
    return "lng_terminal_" + str(int(id))


def create_lng_graph_components(
    prm_liquid_natural_gas_data, prm_liquid_natural_gas_matched_with_pipelines
):

    cols_to_fill = ["operator", "facility_n"]
    nodes = create_nodes(
        prm_liquid_natural_gas_data,
        cols_to_fill,
        lng_item_to_node_id,
        "LngTerminal",
        "LngTerminalID",
    )

    edges = create_edges_for_network_connections(
        prm_liquid_natural_gas_matched_with_pipelines,
        "LngTerminal",
        "LngTerminalID",
        "PipelineNode",
        "PipelineNodeID",
        lng_item_to_node_id,
        coord_to_pipe_key,
        "LNG_TERMINAL_PIPELINE_CONNECTOR",
    )

    return nodes, edges


def match_lng_terminals_with_shipping_routes(
    prm_liquid_natural_gas_data, prm_shipping_routes_data, parameters
):
    starts = pd.DataFrame()
    starts["node_id"] = prm_shipping_routes_data["From Node0"]
    starts["coordinate"] = prm_shipping_routes_data["starting_point"]

    ends = pd.DataFrame()
    ends["node_id"] = prm_shipping_routes_data["To Node0"]
    ends["coordinate"] = prm_shipping_routes_data["end_point"]

    shipping_nodes = pd.concat([starts, ends], ignore_index=True)

    shipping_nodes = shipping_nodes.drop_duplicates(subset=["node_id"])

    shipping_nodes["node_id"] = shipping_nodes.node_id.apply(
        shipping_node_item_to_node_id
    )

    shipping_nodes["point"] = shipping_nodes.coordinate.apply(geometry.Point)

    shipping_nodes.reset_index(drop=True, inplace=True)

    prm_liquid_natural_gas_data[
        "Point"
    ] = prm_liquid_natural_gas_data.coordinates.apply(geometry.Point)

    def _find_nearest_point(port, shipping_nodes):
        nearest = ops.nearest_points(
            port.Point, geometry.MultiPoint(shipping_nodes.point)
        )
        return shipping_nodes[shipping_nodes.point == nearest[1]]["node_id"].values[0]

    nearest_nodes = Parallel(n_jobs=parameters["joblib_n_jobs"])(
        delayed(_find_nearest_point)(port, shipping_nodes)
        for _, port in prm_liquid_natural_gas_data.iterrows()
    )

    prm_liquid_natural_gas_data["nearest_shipping_node"] = nearest_nodes

    return prm_liquid_natural_gas_data


def create_lng_shipping_edges(prm_lng_terminals_matched_with_routes):

    terminal_ship_edge = pd.DataFrame()

    terminal_ship_edge[
        "LngTerminal:START_ID(LngTerminal)"
    ] = prm_lng_terminals_matched_with_routes["item_id"].apply(lng_item_to_node_id)

    terminal_ship_edge[
        "ShipNode:END_ID(ShippingNode)"
    ] = prm_lng_terminals_matched_with_routes["nearest_shipping_node"]

    terminal_ship_edge[":TYPE"] = "LNG_TERMINAL_SHIP_CONNECTOR"

    terminal_ship_edge["impedance"] = 0

    return terminal_ship_edge
