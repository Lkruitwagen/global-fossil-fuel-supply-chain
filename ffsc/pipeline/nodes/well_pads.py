import pandas as pd
from shapely import geometry

from ffsc.pipeline.nodes.utils.geomatching_utils import geodesic_point_buffer
from ffsc.pipeline.nodes.utils import (
    preprocess_geodata,
    create_nodes,
    create_edges_for_network_connections,
)
from ffsc.pipeline.nodes.pipeline_nodes import coord_to_pipe_key


def preprocess_well_pads_data(data):
    df = preprocess_geodata(data)
    return df.rename({"type_x": "pad_type", "type_y": "geometry_type"}, axis=1)


def match_well_pads_to_pipelines(well_pads, pipelines, parameters):
    pipelines = (
        pipelines.apply(
            lambda row: pd.Series(row["coordinates"])
            if row["type"] == "MultiLineString"
            else pd.Series([list(row["coordinates"])]),
            axis=1,
        )
        .stack()
        .reset_index()
        .drop(columns=["level_1"])
        .rename(columns={"level_0": "pipeline_id", 0: "coordinates"})
    )

    flattened = []
    for l in pipelines.coordinates:
        flattened.extend(l)

    mp = geometry.MultiPoint(flattened)

    nearest_pipeline_node = []
    for coord in well_pads.coordinate:
        p = geometry.Point(coord)
        res = geodesic_point_buffer(p.y, p.x, parameters["max_geo_matching_distance"]).intersection(mp)
        nearest_pipeline_node.append(res)

    well_pads["nearest_pipeline_node"] = nearest_pipeline_node


def well_pad_item_to_node_id(id):
    return "well_pad_" + str(int(id))


def create_well_pad_graph_components(
    prm_well_pads_data, prm_well_pads_matched_with_pipelines
):

    cols_to_fill = ["operator", "facility_n"]
    nodes = create_nodes(
        prm_well_pads_data,
        cols_to_fill,
        well_pad_item_to_node_id,
        "WellPad",
        "WellPadID",
    )

    edges = create_edges_for_network_connections(
        prm_well_pads_matched_with_pipelines,
        "WellPad",
        "WellPadID",
        "PipelineNode",
        "PipelineNodeID",
        well_pad_item_to_node_id,
        coord_to_pipe_key,
        "WELL_PAD_PIPELINE_CONNECTOR",
    )

    return nodes, edges
