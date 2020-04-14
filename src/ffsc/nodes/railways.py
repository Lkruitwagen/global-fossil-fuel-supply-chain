from src.ffsc.nodes.intersections import find_intersecting_points
from src.ffsc.nodes.utils import (
    preprocess_geodata,
    unique_nodes_from_segments,
    convert_segments_to_lines,
    calculate_havesine_distance,
)

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
from typing import List, AnyStr


def preprocess_railway_data_int(data):
    df = preprocess_geodata(data, "railway_id")
    df = df.rename({"type_x": "rail_type", "type_y": "geometry_type"}, axis=1)
    # The country for all rows with missing region was either one or several of following:
    # United States of America, Canada, and Mexico
    # or missing.
    df.loc[df.md_country.notnull(), "md_region"] = df.loc[
        df.md_country.notnull(), "md_region"
    ].fillna("N. and C. America")
    return df[
        [
            "railway_id",
            "md_country",
            "md_region",
            "rail_type",
            "geometry_type",
            "coordinates",
        ]
    ]


def preprocess_railway_data_prm(int_railways, parameters):
    railway_df = (
        int_railways[["railway_id", "geometry_type", "coordinates"]]
        .apply(
            lambda row: pd.Series(row["coordinates"])
            if row["geometry_type"] == "MultiLineString"
            else pd.Series([list(row["coordinates"])]),
            axis=1,
        )
        .stack()
        .reset_index()
        .drop(columns=["level_1"])
        .rename(columns={"level_0": "railway_id", 0: "coordinates"})
        .reset_index()
        .rename(columns={"index": "railway_segment_id"})
    )
    railway_df = railway_df.merge(
        int_railways[["railway_id", "md_region", "md_country"]], on="railway_id"
    )
    railway_df["railway_object"] = railway_df["coordinates"].apply(geometry.LineString)

    railway_missing_region_df = railway_df.loc[railway_df.md_region.isna()].copy()

    railway_missing_region_gdf = gpd.GeoDataFrame(
        railway_missing_region_df, geometry=railway_missing_region_df["railway_object"]
    )

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world["buffered_geometry"] = world.geometry.geometry.buffer(10)

    retrieved_region_gdf = gpd.sjoin(
        railway_missing_region_gdf,
        gpd.GeoDataFrame(
            world[["name", "continent"]], geometry=world["buffered_geometry"]
        ),
        how="left",
        op="intersects",
    )

    region_dict = {
        "Oceania": "Australia and Oceania",
        "Africa": "Africa",
        "North America": "N. and C. America",
        "Asia": "Asia",
        "South America": "South America",
        "Europe": "Europe",
    }

    retrieved_region_gdf["retrieved_region"] = retrieved_region_gdf.continent.map(
        region_dict
    )

    retrieved_region_gdf.loc[
        retrieved_region_gdf.retrieved_region == "Asia", "retrieved_region"
    ] = retrieved_region_gdf.loc[
        retrieved_region_gdf.retrieved_region == "Asia", "name"
    ].apply(
        lambda x: "Middle East"
        if x
        in [
            "Iraq",
            "Turkey",
            "Armenia",
            "Azerbaijan",
            "Iran",
            "Kuwait",
            "Israel",
            "Jordan",
            "Syria",
            "Saudi Arabia",
            "Lebanon",
        ]
        else "Asia"
    )

    retrieved_region_gdf.dropna(subset=["retrieved_region"], inplace=True)

    retrieved_region_gdf = (
        retrieved_region_gdf.drop_duplicates(subset=["railway_id", "retrieved_region"])
        .sort_values(["railway_id", "retrieved_region"])
        .groupby("railway_id")
        .retrieved_region.apply(lambda x: "; ".join(x.tolist()))
        .reset_index()
    )

    railway_df = railway_df.merge(retrieved_region_gdf, how="left", on="railway_id")

    railway_df.loc[:, "md_region"] = railway_df.loc[:, "md_region"].fillna(
        railway_df.loc[:, "retrieved_region"]
    )

    railway_df.drop(columns=["retrieved_region"], inplace=True)

    # railway_df.dropna(subset=['md_region'], inplace=True)

    prm_railways_data = (
        railway_df.loc[railway_df.md_region.notnull()]
        .groupby("md_region")
        .apply(
            find_intersecting_points,
            parameters=parameters,
            object_column="railway_object",
            entity_ids=["railway_id", "railway_segment_id"],
        )
        .reset_index()
    )

    prm_railways_data = prm_railways_data[
        ["railway_segment_id", "railway_id", "snapped_geometry"]
    ].copy()

    # Amend the pipelines that did not have any intersections.
    prm_railways_data = pd.concat(
        [
            railway_df.loc[
                ~railway_df.railway_segment_id.isin(
                    prm_railways_data.railway_segment_id.unique()
                ),
                ["railway_segment_id", "railway_id", "railway_object"],
            ].rename(columns={"railway_object": "snapped_geometry"}),
            prm_railways_data,
        ],
        ignore_index=True,
    ).reset_index(drop=True)

    return prm_railways_data


def coord_to_rail_key(coord: List) -> AnyStr:
    return "railway_node_" + "".join([str(item) for item in coord])


def create_railway_graph_components(prm_railways_data):
    unique_nodes = unique_nodes_from_segments(prm_railways_data.snapped_geometry)

    node_df = pd.DataFrame()
    node_df["coordinates"] = unique_nodes
    node_df["RailwayNodeID:ID(RailwayNode)"] = [
        coord_to_rail_key(coord) for coord in unique_nodes
    ]
    node_df["lat"] = [coord[0] for coord in unique_nodes]
    node_df["long"] = [coord[1] for coord in unique_nodes]
    node_df[":LABEL"] = "RailwayNode"

    edges = convert_segments_to_lines(
        [list(line.coords) for line in prm_railways_data.snapped_geometry]
    )

    edge_df = pd.DataFrame()
    edge_df["StartNodeId:START_ID(RailwayNode)"] = [
        coord_to_rail_key(edge[0]) for edge in edges
    ]
    edge_df["EndNodeId:END_ID(RailwayNode)"] = [
        coord_to_rail_key(edge[1]) for edge in edges
    ]
    edge_df[":TYPE"] = "RAILWAY_CONNECTION"

    edge_df["distance"] = calculate_havesine_distance(
        np.array([edge[0] for edge in edges]), np.array([edge[1] for edge in edges])
    )
    edge_df["impedance"] = edge_df["distance"] ** 2

    return node_df, edge_df
