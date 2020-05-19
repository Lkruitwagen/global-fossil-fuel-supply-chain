import ast
from shapely import geometry, ops
import geopandas as gpd
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from ffsc.pipeline.nodes.port_nodes import port_item_to_node_id
from ffsc.pipeline.nodes.railways import coord_to_rail_key
from ffsc.pipeline.nodes.pipeline_nodes import coord_to_pipe_key


def preprocess_city_data_int(raw_cities_energy_data, raw_cities_euclidean_data):
    left_columns = set(raw_cities_energy_data.columns)
    right_columns = set(raw_cities_euclidean_data.columns)
    common_columns = left_columns.intersection(right_columns)
    left_columns = left_columns.difference(common_columns)
    right_columns = right_columns.difference(common_columns)

    output_df = (
        raw_cities_energy_data[list(left_columns.union(common_columns))]
        .merge(
            raw_cities_euclidean_data[right_columns],
            left_index=True,
            right_index=True,
            how="inner",
        )
        .reset_index()
        .rename(columns={"index": "city_id"})
    )

    return output_df


def preprocess_city_data_prm(int_cities_data):
    prm_cities_data = int_cities_data.copy()
    prm_cities_data["city_dict_"] = prm_cities_data["geom_gj"].apply(ast.literal_eval)
    prm_cities_data["city_geometry"] = prm_cities_data["city_dict_"].apply(
        lambda x: geometry.Polygon(x["coordinates"][0])
        if x["type"] == "Polygon"
        else ops.unary_union([geometry.Polygon(y) for y in x["coordinates"][0]])
    )
    prm_cities_data["city_centroid_"] = prm_cities_data["city_geometry"].apply(
        lambda x: x.centroid
    )
    prm_cities_data["city_representative_point_"] = prm_cities_data[
        "city_geometry"
    ].apply(lambda x: x.representative_point())
    prm_cities_data["city_point"] = prm_cities_data.apply(
        lambda row: row["city_centroid_"]
        if row["city_centroid_"].within(row["city_geometry"])
        else row["city_representative_point_"],
        axis=1,
    )
    prm_cities_data.rename(columns={"geometry": "euclidean_geom"}, inplace=True)
    prm_cities_data.drop(
        columns=["city_dict_", "city_centroid_", "city_representative_point_"],
        inplace=True,
    )
    return prm_cities_data


def match_cities_with_pipelines_and_railways(
    prm_cities_data, prm_pipelines_data, prm_railways_data, prm_ports_data, parameters
):
    cities = prm_cities_data[
        ["city_id", "euclidean_geom", "city_geometry", "city_point"]
    ]
    cities_gdf = gpd.GeoDataFrame(cities, geometry=cities["euclidean_geom"])
    pipelines_gdf = gpd.GeoDataFrame(
        prm_pipelines_data[["pipeline_segment_id", "snapped_geometry"]],
        geometry=prm_pipelines_data["snapped_geometry"],
    )
    railways_gdf = gpd.GeoDataFrame(
        prm_railways_data[["railway_segment_id", "snapped_geometry"]],
        geometry=prm_pipelines_data["snapped_geometry"],
    )
    ports = prm_ports_data[["item_id", "coordinates"]]
    ports["coordinates"] = ports["coordinates"].apply(geometry.Point)
    ports_gdf = gpd.GeoDataFrame(ports, geometry=ports["coordinates"])

    intersecting_cities_pipelines = find_cities_intersections(
        cities_gdf, pipelines_gdf, "pipeline", parameters
    )
    intersecting_cities_railways = find_cities_intersections(
        cities_gdf, railways_gdf, "railway", parameters
    )
    cities_gdf = gpd.GeoDataFrame(cities, geometry=cities["city_geometry"])
    intersecting_cities_ports = gpd.sjoin(
        cities_gdf, ports_gdf, how="inner", op="intersects"
    )
    intersecting_cities_ports["connected_node_type"] = "port"
    intersecting_cities_ports = intersecting_cities_ports[
        ["city_id", "city_point", "item_id", "coordinates", "connected_node_type"]
    ].rename(
        columns={
            "item_id": "connected_segment_id",
            "coordinates": "connected_node_point",
        }
    )

    intersecting_cities = pd.concat(
        [
            intersecting_cities_pipelines,
            intersecting_cities_railways,
            intersecting_cities_ports,
        ],
        ignore_index=True,
    )

    return intersecting_cities


def find_cities_intersections(cities, network, entity, parameters):
    cities_with_intersections = gpd.sjoin(cities, network, how="inner", op="intersects")

    def _find_nearest_point(city, entity_shape):
        shape_points = geometry.MultiPoint(
            [geometry.Point(point) for point in entity_shape.coords]
        )
        return ops.nearest_points(city, shape_points)[1]

    nearest_entities = Parallel(n_jobs=parameters["joblib_n_jobs"])(
        delayed(_find_nearest_point)(row["city_point"], row["snapped_geometry"])
        for _, row in cities_with_intersections.iterrows()
    )

    cities_with_intersections["connected_node_point"] = nearest_entities
    cities_with_intersections["connected_node_type"] = entity

    return cities_with_intersections[
        [
            "city_id",
            "city_point",
            entity + "_segment_id",
            "connected_node_point",
            "connected_node_type",
        ]
    ].rename(columns={entity + "_segment_id": "connected_segment_id"})


def city_item_to_node_id(id):
    return "city_node_" + str(int(id))


def create_city_node_table(prm_cities_data, parameters):
    city_nodes = prm_cities_data[["city_id"] + parameters["cities_columns"]].copy()

    for commodity in ["oil", "coal", "gas"]:
        city_nodes[f"total_{commodity}_consumption"] = np.abs(
            city_nodes[
                [
                    col
                    for col in city_nodes.columns
                    if col[-(len(commodity) + 1) :] == f"_{commodity}"
                ]
            ]
        ).sum(axis=1)

    city_nodes[":LABEL"] = "CityNode"
    city_nodes["CityNodeId:ID(CityNode)"] = city_nodes.city_id.apply(
        city_item_to_node_id
    )
    return city_nodes


def create_city_pipeline_edges(cities_intersecting):
    cities_intersecting_pipelines = cities_intersecting.loc[
        cities_intersecting.connected_node_type == "pipeline"
    ]
    city_pipe_edge_frame = pd.DataFrame()

    city_pipe_edge_frame[
        "PipelineNode:START_ID(PipelineNode)"
    ] = cities_intersecting_pipelines["connected_node_point"].apply(
        lambda x: coord_to_pipe_key(x.coords[0])
    )

    city_pipe_edge_frame["CityNode:END_ID(CityNode)"] = cities_intersecting_pipelines[
        "city_id"
    ].apply(city_item_to_node_id)

    city_pipe_edge_frame[":TYPE"] = "CITY_PIPELINE_CONNECTOR"

    city_pipe_edge_frame["impedance"] = 0

    return city_pipe_edge_frame


def create_city_railway_edges(cities_intersecting):
    cities_intersecting_railways = cities_intersecting.loc[
        cities_intersecting.connected_node_type == "railway"
    ]
    city_rail_edge_frame = pd.DataFrame()

    city_rail_edge_frame[
        "RailwayNode:START_ID(RailwayNode)"
    ] = cities_intersecting_railways["connected_node_point"].apply(
        lambda x: coord_to_rail_key(x.coords[0])
    )

    city_rail_edge_frame["CityNode:END_ID(CityNode)"] = cities_intersecting_railways[
        "city_id"
    ].apply(city_item_to_node_id)

    city_rail_edge_frame[":TYPE"] = "CITY_RAILWAY_CONNECTOR"

    city_rail_edge_frame["impedance"] = 0

    return city_rail_edge_frame


def create_city_port_edges(cities_intersecting):
    cities_intersecting_ports = cities_intersecting.loc[
        cities_intersecting.connected_node_type == "port"
    ]
    city_port_edge_frame = pd.DataFrame()

    city_port_edge_frame["PortNode:START_ID(PortNode)"] = cities_intersecting_ports[
        "connected_segment_id"
    ].apply(port_item_to_node_id)

    city_port_edge_frame["CityNode:END_ID(CityNode)"] = cities_intersecting_ports[
        "city_id"
    ].apply(city_item_to_node_id)

    city_port_edge_frame[":TYPE"] = "CITY_PORT_CONNECTOR"

    city_port_edge_frame["impedance"] = 0

    return city_port_edge_frame
