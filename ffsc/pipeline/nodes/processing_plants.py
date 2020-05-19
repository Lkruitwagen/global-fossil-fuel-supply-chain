from ffsc.pipeline.nodes.utils import (
    preprocess_geodata,
    create_nodes,
    create_edges_for_network_connections,
)
from ffsc.pipeline.nodes.pipeline_nodes import coord_to_pipe_key


def preprocess_processing_plants_data(data):
    df = preprocess_geodata(data)
    return df.rename({"type_x": "facility_type", "type_y": "geometry_type"}, axis=1)


def processing_plant_item_to_node_id(id):
    return "processing_plant_" + str(int(id))


def create_processing_plant_graph_component(
    prm_processing_plants_data, prm_processing_plants_matched_with_pipelines
):

    cols_to_fill = ["operator"]
    nodes = create_nodes(
        prm_processing_plants_data,
        cols_to_fill,
        processing_plant_item_to_node_id,
        "ProcessingPlant",
        "ProcessingPlantID",
    )

    edges = create_edges_for_network_connections(
        prm_processing_plants_matched_with_pipelines,
        "ProcessingPlant",
        "ProcessingPlantID",
        "PipelineNode",
        "PipelineNodeID",
        processing_plant_item_to_node_id,
        coord_to_pipe_key,
        "PROCESSING_PLANT_PIPELINE_CONNECTOR",
    )

    return nodes, edges
