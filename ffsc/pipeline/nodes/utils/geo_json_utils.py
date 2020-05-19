import pandas as pd
from typing import Dict, AnyStr


def preprocess_geodata(data: Dict, index_name: AnyStr = "item_id") -> pd.DataFrame:
    """
    Loads a geojson file into a pandas dataframe.
    We found this method to be faster and more robust than loading the file with geopandas,
    especially if the file is large.

    :param data: A dictionary representing the geojson file
    :param index_name: The name to be given to the index column
    :return: A pandas dataframe containing all data from the geojson
    """
    features_df = (
        pd.DataFrame([item["properties"] for item in data["features"]])
        .reset_index()
        .rename(columns={"index": index_name})
    )

    geometry_df = (
        pd.DataFrame([item["geometry"] for item in data["features"]])
        .reset_index()
        .rename(columns={"index": index_name})
    )

    return features_df.merge(geometry_df, on=[index_name])
