import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
from src.ffsc.nodes.utils import (
    preprocess_geodata,
    create_nodes,
    create_edges_for_network_connections,
    unique_nodes_from_segments,
)
from src.ffsc.nodes.pipeline_nodes import coord_to_pipe_key


def preprocess_oil_field_data(data):
    df = preprocess_geodata(data)
    return df.rename({"type_x": "field_type", "type_y": "geometry_type"}, axis=1)


def oil_field_item_to_node_id(id):
    return "oil_field_" + str(int(id))


def merge_oil_fields_with_pipeline_network(network_data, facility_data, parameters):

    polys = facility_data.loc[facility_data.geometry_type == "Polygon", "coordinates"]
    polys = [Polygon(coord[0]) for coord in polys]
    facility_data.loc[facility_data.geometry_type == "Polygon", "geometry"] = polys

    multi_polys = facility_data.loc[
        facility_data.geometry_type == "MultiPolygon", "coordinates"
    ]
    multi_polys = [
        MultiPolygon([Polygon(coord[0]) for coord in coords]) for coords in multi_polys
    ]
    facility_data.loc[
        facility_data.geometry_type == "MultiPolygon", "geometry"
    ] = multi_polys

    facility_data["centroid"] = [obj.centroid for obj in facility_data["geometry"]]

    unique_nodes = unique_nodes_from_segments(network_data.snapped_geometry)

    node_df = pd.DataFrame()
    node_df["coordinates"] = unique_nodes
    node_df["geometry"] = node_df.coordinates.apply(Point)

    geo_refineries = gpd.GeoDataFrame(facility_data)

    geo_pipelines = gpd.GeoDataFrame(node_df)

    matched = gpd.sjoin(geo_refineries, geo_pipelines, op="contains", how="inner")

    matched = pd.DataFrame(matched)

    matched = matched.rename({"coordinates_right": "network_coordinates"}, axis=1)

    matched["facility_coordinates"] = matched["centroid"].apply(lambda x: x.coords[0])

    return matched


def create_oil_field_graph_component(
    prm_oil_field_data, prm_oil_field_matched_with_pipelines
):

    cols_to_fill = [
        "facility_n",
        "status",
        "field_type",
        "md_country",
        "md_source",
        "md_region",
        "commodity",
    ]
    nodes = create_nodes(
        prm_oil_field_data,
        cols_to_fill,
        oil_field_item_to_node_id,
        "OilField",
        "OilFieldID",
    )

    edges = create_edges_for_network_connections(
        prm_oil_field_matched_with_pipelines,
        "OilField",
        "OilFieldID",
        "PipelineNode",
        "PipelineNodeID",
        oil_field_item_to_node_id,
        coord_to_pipe_key,
        "OIL_FIELD_PIPELINE_CONNECTOR",
    )

    return nodes, edges
