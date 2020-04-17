import pandas as pd

from shapely import geometry, ops
from joblib import Parallel, delayed
from src.ffsc.nodes.pipeline_nodes import coord_to_pipe_key
from src.ffsc.nodes.railways import coord_to_rail_key

from src.ffsc.nodes.utils import preprocess_geodata

from src.ffsc.nodes.shipping_nodes import shipping_node_item_to_node_id


def preprocess_port_data(data):

    return preprocess_geodata(data).rename(
        {"type_x": "port_type", "type_y": "type"}, axis=1
    )


def match_ports_with_shipping_routes(ports, prm_shipping_routes_data, parameters):
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

    ports["Point"] = ports.coordinates.apply(geometry.Point)

    def _find_nearest_point(port, shipping_nodes):
        nearest = ops.nearest_points(
            port.Point, geometry.MultiPoint(shipping_nodes.point)
        )
        return shipping_nodes[shipping_nodes.point == nearest[1]]["node_id"].values[0]

    nearest_nodes = Parallel(n_jobs=parameters["joblib_n_jobs"])(
        delayed(_find_nearest_point)(port, shipping_nodes)
        for _, port in ports.iterrows()
    )

    ports["nearest_shipping_node"] = nearest_nodes

    return ports


def create_port_node_table(prm_ports_data: pd.DataFrame):

    port_node_dataframe = prm_ports_data.rename(
        {"item_id": "PortNodeId:ID(PortNode)"}, axis=1
    ).drop("type", axis=1)
    port_node_dataframe[":LABEL"] = "PortNode"
    port_node_dataframe["PortNodeId:ID(PortNode)"] = [
        "port_node_" + str(port_id)
        for port_id in port_node_dataframe["PortNodeId:ID(PortNode)"]
    ]

    port_node_dataframe["md_region"] = port_node_dataframe["md_region"].fillna(
        "Unknown"
    )

    port_node_dataframe = port_node_dataframe.dropna(axis=1)

    port_node_dataframe["facility_n"] = port_node_dataframe["facility_n"].str.replace(
        "\W", ""
    )

    return port_node_dataframe


def port_item_to_node_id(id):
    return "port_node_" + str(int(id))


def create_port_ship_edges(prm_ports_matched_with_routes):

    port_ship_edge_frame = pd.DataFrame()

    port_ship_edge_frame["PortNode:START_ID(PortNode)"] = prm_ports_matched_with_routes[
        "item_id"
    ].apply(port_item_to_node_id)

    port_ship_edge_frame[
        "ShipNode:END_ID(ShippingNode)"
    ] = prm_ports_matched_with_routes["nearest_shipping_node"]

    port_ship_edge_frame[":TYPE"] = "PORT_SHIP_CONNECTOR"

    port_ship_edge_frame["impedance"] = 0

    return port_ship_edge_frame


def create_port_pipeline_edges(prm_ports_matched_with_pipelines):
    edges = prm_ports_matched_with_pipelines[["item_id", "network_coordinates"]]

    edges = edges.rename({"item_id": "PortNode:START_ID(PortNode)"}, axis=1)
    edges["PortNode:START_ID(PortNode)"] = edges["PortNode:START_ID(PortNode)"].apply(
        port_item_to_node_id
    )

    edges["PipelineNodeID:END_ID(PipelineNode)"] = edges["network_coordinates"].apply(
        coord_to_pipe_key
    )

    edges = edges.drop("network_coordinates", axis=1)

    edges[":TYPE"] = "PORT_PIPELINE_CONNECTOR"

    return edges


def create_port_railway_edges(prm_ports_matched_with_railways):
    edges = prm_ports_matched_with_railways[["item_id", "network_coordinates"]]

    edges = edges.rename({"item_id": "PortNode:START_ID(PortNode)"}, axis=1)
    edges["PortNode:START_ID(PortNode)"] = edges["PortNode:START_ID(PortNode)"].apply(
        port_item_to_node_id
    )

    edges["RailwayNodeID:END_ID(PipelineNode)"] = edges["network_coordinates"].apply(
        coord_to_rail_key
    )

    edges = edges.drop("network_coordinates", axis=1)

    edges[":TYPE"] = "PORT_RAILWAY_CONNECTOR"

    return edges
