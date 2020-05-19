import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import transform
from typing import Dict
from functools import partial
import pyproj
from .python_helpers import unique_nodes_from_segments

proj_wgs84 = pyproj.Proj(init='epsg:4326')

def merge_facility_with_transportation_network_graph(
    network_data: pd.DataFrame, facility_data: pd.DataFrame, parameters: Dict
) -> pd.DataFrame:
    """
    Merges facility data (e.g. refineries) with network data (pipelines or railways).
    A maximum distance given in the parameters is used. This means, one facility can be linked to many network nodes
    and one network node can be linked to many facilities.
    :param network_data: A dataframe containing line segments that form a network.
    We assume that the column containing the line segments is called "snapped_geometry"
    :param facility_data: A dataframe containing facilities as points.
    We assume that the facility location is present as a list of coordinates ina  column called coordinates
    :param parameters: Parameters from the kedro parameters.yml file. We assume it contains a max_geo_matching_distance
    :return: Dataframe facilities merged with network data.
    Retains all attributes of the facilities but only the location of network nodes.
    """

    facility_data["coord_x_"] = facility_data.coordinates.apply(lambda x: x[0])
    facility_data["coord_y_"] = facility_data.coordinates.apply(lambda x: x[1])
    facility_data = facility_data.loc[(facility_data.coord_x_ <= 180) &
                                      (facility_data.coord_x_ >= -180) &
                                      (facility_data.coord_y_ <= 90) &
                                      (facility_data.coord_y_ >= -90)]

    facility_data.drop(columns=["coord_x_", "coord_y_"], inplace=True)

    facility_data["geometry"] = facility_data.coordinates.apply(Point)

    facility_data["geometry"] = facility_data.geometry.apply(
        lambda x: geodesic_point_buffer(x.y, x.x, parameters["max_geo_matching_distance"])
    )

    unique_nodes = unique_nodes_from_segments(network_data.snapped_geometry)

    node_df = pd.DataFrame()
    node_df["coordinates"] = unique_nodes
    node_df["geometry"] = node_df.coordinates.apply(Point)

    geo_refineries = gpd.GeoDataFrame(facility_data)

    geo_pipelines = gpd.GeoDataFrame(node_df)

    matched = gpd.sjoin(geo_refineries, geo_pipelines, op="contains", how="inner")

    matched = pd.DataFrame(matched)

    matched = matched.rename(
        {
            "coordinates_left": "facility_coordinates",
            "coordinates_right": "network_coordinates",
        },
        axis=1,
    )

    return matched


def geodesic_point_buffer(lat, lon, km):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    return Polygon(transform(project, buf).exterior.coords[:])
