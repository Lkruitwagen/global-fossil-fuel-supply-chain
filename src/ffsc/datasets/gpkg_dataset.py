from kedro.io.core import AbstractDataSet
from pathlib import Path
from typing import Any, Dict
import geopandas as gpd
import pandas as pd


class GPKGDataSet(AbstractDataSet):
    def __init__(self, dirpath: str, filelistpath: str):
        self.dir_path = dirpath
        self.file_list_path = filelistpath
        super().__init__()

    def _save(self):
        raise NotImplementedError

    def _load(self) -> Any:
        load_file_list_path = Path(self.file_list_path)
        entity_list = pd.read_csv(load_file_list_path.open("rb")).dropna(
            subset=["Code"]
        )
        all_entities_list = []
        for _, country in entity_list.iterrows():
            load_path = Path(self.dir_path + "/" + str(country["Code"]) + ".gpkg")
            if load_path.is_file():
                with load_path.open("rb") as local_file:
                    all_entities_list.append(gpd.read_file(local_file))

        all_entities_data = pd.concat(all_entities_list, ignore_index=True)
        # load_path = Path(self.file_path)
        # with load_path.open("rb") as local_file:
        #     return gpd.read_file(local_file)
        return all_entities_data

    def _describe(self) -> Dict[str, Any]:
        return "GPKG Dataset."
