import geopandas as gpd
import pandas as pd
from shapely import ops


def find_intersecting_points(
    network_df,
    parameters,
    object_column="pipeline_object",
    entity_ids=["pipeline_id", "pipeline_segment_id"],
):
    """
    This function finds the intersecting LineString objects, finds the intersecting points, and then snap the
    intersecing points to the LineString objects.

    :param network_df: The dataframe with the LineString objects
    :param parameters: Dictionary with pipeline parameters.
    :param object_column: The column in the network_df containing the LineString objects.
    :param entity_ids: The unique keys of the dataframe.
    :return: Returns the dataframe with the LineString objects modified to include the intersecting points.
    """
    # Set up the GeoPandas dataframe
    network_gdf = gpd.GeoDataFrame(network_df, geometry=network_df[object_column])
    network_gdf.sindex

    # Find the intersecting LineStrings
    intersected_gdf = gpd.sjoin(
        network_gdf[entity_ids + ["geometry"]],
        network_gdf[entity_ids + ["geometry"]],
        op="intersects",
    )

    # Bring in the LineString object of the other intersecting object:
    intersected_gdf = (
        intersected_gdf.merge(
            network_gdf[["geometry"]], left_on="index_right", right_index=True
        )
        .reset_index(drop=True)
        .rename(columns={"geometry_x": "geometry_left", "geometry_y": "geometry_right"})
    )

    # Find the intersection of the intersecting LineStrings:
    intersected_gdf["intersection"] = intersected_gdf.apply(
        lambda row: row["geometry_left"].intersection(row["geometry_right"]), axis=1
    )

    # We only focus on the intersection where the type is either Point or MultiPoint.
    # This removes the intersection of LineStrings with themselves.
    # orig_index is the index for intersection.
    points_df = (
        intersected_gdf.loc[
            intersected_gdf.intersection.apply(lambda x: "Point" in x.type)
        ]
        .intersection.apply(
            lambda x: pd.Series(list(x))
            if x.type == "MultiPoint"
            else pd.Series(list([x]))
        )
        .stack()
        .reset_index()
        .drop(columns=["level_1"])
        .rename(columns={"level_0": "orig_index", 0: "intersection_point"})
    )

    # Remove the duplicated intersections (points appear twice per intersection.)
    points_gdf = gpd.GeoDataFrame(
        points_df["orig_index"], geometry=points_df["intersection_point"]
    ).drop_duplicates()

    # Bring in the entity_ids and geometries of LinesStrings corresponding for each intersection.
    intersection_df = (
        points_gdf.merge(
            intersected_gdf[
                [
                    entity_id + "_" + direction
                    for direction in ["left", "right"]
                    for entity_id in entity_ids + ["geometry"]
                ]
            ],
            left_on="orig_index",
            right_index=True,
        )
        .rename(columns={"geometry": "intersection_point"})
        .drop(columns=["orig_index"])
    ).reset_index(drop=True)

    # Put the left and right LineStrings on top of each other.
    intersection_df = (
        (
            pd.concat(
                [
                    intersection_df[
                        ["intersection_point"]
                        + [
                            entity_id + "_left"
                            for entity_id in ["geometry"] + entity_ids
                        ]
                    ].rename(
                        columns={
                            entity_id + "_left": entity_id
                            for entity_id in ["geometry"] + entity_ids
                        }
                    ),
                    intersection_df[
                        ["intersection_point"]
                        + [
                            entity_id + "_right"
                            for entity_id in ["geometry"] + entity_ids
                        ]
                    ].rename(
                        columns={
                            entity_id + "_right": entity_id
                            for entity_id in ["geometry"] + entity_ids
                        }
                    ),
                ],
                ignore_index=True,
            )
        )
        .reset_index(drop=True)
        .drop_duplicates()
    )

    # For each LineString, combine all intersecting points to a single MultiPoint object.

    intersecting_points = (
        intersection_df.groupby(entity_ids)
        .intersection_point.apply(lambda x: gpd.GeoDataFrame(geometry=x).unary_union)
        .reset_index()
    )

    # Bring in entity ids and geometries of intersecting LineStrings to the objects above.
    intersecting_points = intersecting_points.merge(
        intersection_df[entity_ids + ["geometry"]].drop_duplicates(), on=entity_ids
    )

    # Snap the intersecting points to LineString objects.
    intersecting_points["snapped_geometry"] = intersecting_points.apply(
        lambda row: ops.snap(
            row["geometry"], row["intersection_point"], parameters["snapping_threshold"]
        ),
        axis=1,
    )

    return intersecting_points
