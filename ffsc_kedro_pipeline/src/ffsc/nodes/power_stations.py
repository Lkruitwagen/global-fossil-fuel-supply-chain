from src.ffsc.nodes.utils import (
    preprocess_geodata,
    create_nodes,
    create_edges_for_network_connections,
)
from src.ffsc.nodes.pipeline_nodes import coord_to_pipe_key
from src.ffsc.nodes.railways import coord_to_rail_key


def preprocess_power_stations_data(data):
    return preprocess_geodata(data)


def power_station_item_to_node_id(id):
    return "power_station_" + str(int(id))


def create_power_station_graph_components(
    prm_power_stations_data,
    prm_power_stations_data_matched_with_pipelines,
    prm_power_stations_data_matched_with_railways,
):

    cols_to_fill = ["name", "fuel1", "owner"]
    nodes = create_nodes(
        prm_power_stations_data,
        cols_to_fill,
        power_station_item_to_node_id,
        "PowerStation",
        "PowerStationID",
    )

    pipeline_edges = create_edges_for_network_connections(
        prm_power_stations_data_matched_with_pipelines,
        "PowerStation",
        "PowerStationID",
        "PipelineNode",
        "PipelineNodeID",
        power_station_item_to_node_id,
        coord_to_pipe_key,
        "POWER_STATION_PIPELINE_CONNECTOR",
    )

    railway_edges = create_edges_for_network_connections(
        prm_power_stations_data_matched_with_railways,
        "PowerStation",
        "PowerStationID",
        "RailwayNode",
        "RailwayNodeID",
        power_station_item_to_node_id,
        coord_to_rail_key,
        "POWER_STATION_RAILWAY_CONNECTOR",
    )

    return nodes, pipeline_edges, railway_edges
