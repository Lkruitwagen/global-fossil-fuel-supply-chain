# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Construction of the master pipeline.
"""

from typing import Dict
from kedro.pipeline import Pipeline, node


###########################################################################
# Here you can find an example pipeline, made of two modular pipelines.
#
# Delete this when you start working on your own Kedro project as
# well as pipelines/data_science AND pipelines/data_engineering
# -------------------------------------------------------------------------

from ffsc.pipeline.nodes.city_nodes import (
    preprocess_city_data_int,
    preprocess_city_data_prm,
    match_cities_with_pipelines_and_railways,
    create_city_node_table,
    create_city_pipeline_edges,
    create_city_railway_edges,
    create_city_port_edges,
)
from ffsc.pipeline.nodes.railways import (
    preprocess_railway_data_int,
    preprocess_railway_data_prm,
)
from ffsc.pipeline.nodes.railways import (
    preprocess_railway_data_int,
    preprocess_railway_data_prm,
)
from ffsc.pipeline.nodes import (
    preprocess_shipping_data,
    preprocess_port_data,
    match_ports_with_shipping_routes,
    preprocess_pipeline_data_prm,
    preprocess_pipeline_data_int,
    preprocess_coal_mine_data,
    preprocess_oil_field_data,
    preprocess_lng_data,
    preprocess_power_stations_data,
    preprocess_processing_plants_data,
    preprocess_refineries_data,
    preprocess_well_pads_data,
    create_pipeline_graph_tables,
    create_shipping_graph_tables,
    create_port_node_table,
    create_port_ship_edges,
    create_refinery_graph_components,
    create_oil_field_graph_component,
    create_well_pad_graph_components,
    create_lng_graph_components,
    create_processing_plant_graph_component,
    create_power_station_graph_components,
    create_port_pipeline_edges,
    create_port_railway_edges,
    create_railway_graph_components,
    match_lng_terminals_with_shipping_routes,
    create_coal_mine_graph_components,
    create_lng_shipping_edges,
    merge_oil_fields_with_pipeline_network,
)

from ffsc.pipeline.nodes.utils import merge_facility_with_transportation_network_graph



def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    data_science_pipeline = (
        preprocess_pipeline()
        + geomatching_pipeline()
        + graph_writing_pipeline()
        # + simplify_pipeline() <- todo
    )

    return {"ds": data_science_pipeline, "__default__": data_science_pipeline}


def preprocess_pipeline(**kwargs):
    tags = ["preprocess"]
    return Pipeline(
        [
            node(
                preprocess_shipping_data,
                "raw_shipping_routes_data",
                "prm_shipping_routes_data",
                tags=tags + ["preprocess_shipping"],
            ),
            node(
                preprocess_port_data,
                "raw_ports_data",
                "prm_ports_data",
                tags=tags + ["preprocess_port"],
            ),
            node(
                preprocess_pipeline_data_int,
                "raw_pipelines_data",
                "int_pipelines_data",
                tags=tags + ["preprocess_pipelines"],
            ),
            node(
                preprocess_pipeline_data_prm,
                ["int_pipelines_data", "parameters"],
                "prm_pipelines_data",
                tags=tags + ["preprocess_pipelines"],
            ),
            node(
                preprocess_coal_mine_data,
                "raw_coal_mines_data",
                "prm_coal_mine_data",
                tags=tags + ["preprocess_coal_mines"],
            ),
            node(
                preprocess_oil_field_data,
                "raw_oil_field_data",
                "prm_oil_field_data",
                tags=tags + ["preprocess_oil_fields"],
            ),
            node(
                preprocess_lng_data,
                "raw_liquid_natural_gas_data",
                "prm_liquid_natural_gas_data",
                tags=tags + ["preprocess_lng"],
            ),
            node(
                preprocess_power_stations_data,
                "raw_power_stations_data",
                "prm_power_stations_data",
                tags=tags + ["preprocess_power_stations"],
            ),
            node(
                preprocess_processing_plants_data,
                "raw_processing_plants_data",
                "prm_processing_plants_data",
                tags=tags + ["preprocess_processing_plants"],
            ),
            node(
                preprocess_railway_data_int,
                "raw_railways_data",
                "int_railways_data",
                tags=tags + ["preprocess_railways"],
            ),
            node(
                preprocess_railway_data_prm,
                ["int_railways_data", "parameters"],
                "prm_railways_data",
                tags=tags + ["preprocess_railways"],
            ),
            node(
                preprocess_refineries_data,
                "raw_refineries_data",
                "prm_refineries_data",
                tags=tags + ["preprocess_refineries"],
            ),
            node(
                preprocess_well_pads_data,
                "raw_well_pads_data",
                "prm_well_pads_data",
                tags=tags + ["preprocess_well_pads"],
            ),
            node(
                preprocess_city_data_int,
                ["raw_cities_energy_data", "raw_cities_euclidean_data"],
                "int_cities_data",
                tags=tags + ["preprocess_cities_data"],
            ),
            node(
                preprocess_city_data_prm,
                "int_cities_data",
                "prm_cities_data",
                tags=tags + ["preprocess_cities_data"],
            ),
        ]
    )


def geomatching_pipeline(**kwargs):
    tags = ["geo_matching"]

    return Pipeline(
        [
            node(
                match_ports_with_shipping_routes,
                ["prm_ports_data", "prm_shipping_routes_data", "parameters"],
                "prm_ports_matched_with_routes",
                tags=tags + ["match_ports_with_shipping_routes"],
            ),
            node(
                match_cities_with_pipelines_and_railways,
                [
                    "prm_cities_data",
                    "prm_pipelines_data",
                    "prm_railways_data",
                    "prm_ports_data",
                    "parameters",
                ],
                "prm_cities_matched_with_pipelines_and_railways",
                tags=tags + ["match_ports_with_shipping_routes"],
            ),
            node(
                match_lng_terminals_with_shipping_routes,
                [
                    "prm_liquid_natural_gas_data",
                    "prm_shipping_routes_data",
                    "parameters",
                ],
                "prm_lng_terminals_matched_with_routes",
                tags=tags + ["match_lng_terminals_with_shipping_routes"],
            ),
            node(
                merge_oil_fields_with_pipeline_network,
                ["prm_pipelines_data", "prm_oil_field_data", "parameters"],
                "prm_oil_field_matched_with_pipelines",
                tags=tags + ["oil_field_pipeline_matching"],
            ),
            node(
                merge_facility_with_transportation_network_graph,
                ["prm_pipelines_data", "prm_well_pads_data", "parameters"],
                "prm_well_pads_matched_with_pipelines",
                tags=tags + ["well_pad_pipeline_matching"],
            ),
            node(
                merge_facility_with_transportation_network_graph,
                ["prm_pipelines_data", "prm_refineries_data", "parameters"],
                "prm_refineries_matched_with_pipelines",
                tags=tags + ["match_refineries_with_pipelines"],
            ),
            node(
                merge_facility_with_transportation_network_graph,
                ["prm_pipelines_data", "prm_ports_data", "parameters"],
                "prm_ports_matched_with_pipelines",
                tags=tags + ["port_pipeline_matching"],
            ),
            node(
                merge_facility_with_transportation_network_graph,
                ["prm_railways_data", "prm_ports_data", "parameters"],
                "prm_ports_matched_with_railways",
                tags=tags + ["port_railway_matching"],
            ),
            node(
                merge_facility_with_transportation_network_graph,
                ["prm_pipelines_data", "prm_liquid_natural_gas_data", "parameters"],
                "prm_liquid_natural_gas_matched_with_pipelines",
                tags=tags + ["liquid_natural_gas_pipeline_matching"],
            ),
            node(
                merge_facility_with_transportation_network_graph,
                ["prm_pipelines_data", "prm_processing_plants_data", "parameters"],
                "prm_processing_plants_matched_with_pipelines",
                tags=tags + ["processing_plants_pipeline_matching"],
            ),
            node(
                merge_facility_with_transportation_network_graph,
                ["prm_railways_data", "prm_power_stations_data", "parameters"],
                "prm_power_stations_data_matched_with_railways",
                tags=tags + ["power_stations_railway_matching"],
            ),
            node(
                merge_facility_with_transportation_network_graph,
                ["prm_pipelines_data", "prm_power_stations_data", "parameters"],
                "prm_power_stations_data_matched_with_pipelines",
                tags=tags + ["power_stations_pipeline_matching"],
            ),
            node(
                merge_facility_with_transportation_network_graph,
                ["prm_railways_data", "prm_coal_mine_data", "parameters"],
                "prm_coal_mines_merged_with_railways",
                tags=tags + ["coal_mine_railway_matching"],
            ),
            node(
                merge_facility_with_transportation_network_graph,
                ["prm_railways_data", "prm_ports_data", "parameters"],
                "prm_ports_merged_with_railways",
                tags=tags + ["port_railway_matching"],
            ),
        ]
    )


def graph_writing_pipeline(**kwargs):
    tags = ["graph_writing"]

    return Pipeline(
        [
            node(
                create_shipping_graph_tables,
                "prm_shipping_routes_data",
                ["shipping_node_dataframe", "shipping_edge_dataframe"],
                tags=tags + ["ship_route_graph"],
            ),
            node(
                create_port_node_table,
                "prm_ports_data",
                "port_node_dataframe",
                tags=tags + ["port_nodes"],
            ),
            node(
                create_port_ship_edges,
                "prm_ports_matched_with_routes",
                "port_ship_edge_dataframe",
                tags=tags + ["port_ship_edges"],
            ),
            node(
                create_lng_shipping_edges,
                "prm_lng_terminals_matched_with_routes",
                "lng_shipping_route_edge_dataframe",
                tags=tags + ["lng_ship_edges"],
            ),
            node(
                create_port_pipeline_edges,
                "prm_ports_matched_with_pipelines",
                "port_pipeline_edge_dataframe",
                tags=tags + ["port_pipeline_edges"],
            ),
            node(
                create_port_railway_edges,
                "prm_ports_matched_with_railways",
                "port_railway_edge_dataframe",
                tags=tags + ["port_railway_edges"],
            ),
            node(
                create_pipeline_graph_tables,
                ["prm_pipelines_data", "parameters"],
                ["pipeline_node_dataframe", "pipeline_edge_dataframe"],
                tags=tags + ["pipeline_graph"],
            ),
            node(
                create_refinery_graph_components,
                ["prm_refineries_data", "prm_refineries_matched_with_pipelines"],
                ["refinery_nodes_dataframe", "refinery_pipeline_edge_dataframe"],
                tags=tags + ["refinery_graph"],
            ),
            node(
                create_oil_field_graph_component,
                ["prm_oil_field_data", "prm_oil_field_matched_with_pipelines"],
                ["oil_field_nodes_dataframe", "oil_field_edge_dataframe"],
                tags=tags + ["oil_field_graph"],
            ),
            node(
                create_well_pad_graph_components,
                ["prm_well_pads_data", "prm_well_pads_matched_with_pipelines"],
                ["well_pad_nodes_dataframe", "well_pad_pipeline_edge_dataframe"],
                tags=tags + ["well_pad_graph"],
            ),
            node(
                create_lng_graph_components,
                [
                    "prm_liquid_natural_gas_data",
                    "prm_liquid_natural_gas_matched_with_pipelines",
                ],
                ["lng_nodes_dataframe", "lng_pipeline_edge_dataframe"],
                tags=tags + ["lng_graph"],
            ),
            node(
                create_processing_plant_graph_component,
                [
                    "prm_processing_plants_data",
                    "prm_processing_plants_matched_with_pipelines",
                ],
                [
                    "processing_plant_nodes_dataframe",
                    "processing_plant_pipeline_edge_dataframe",
                ],
                tags=tags + ["processing_plant_graph"],
            ),
            node(
                create_power_station_graph_components,
                [
                    "prm_power_stations_data",
                    "prm_power_stations_data_matched_with_pipelines",
                    "prm_power_stations_data_matched_with_railways",
                ],
                [
                    "power_station_nodes_dataframe",
                    "power_station_pipeline_edge_dataframe",
                    "power_station_railway_edge_dataframe",
                ],
                tags=tags + ["power_station_graph"],
            ),
            node(
                create_railway_graph_components,
                "prm_railways_data",
                ["railway_nodes_dataframe", "railway_edge_dataframe"],
                tags=tags + ["railway_graph"],
            ),
            node(
                create_coal_mine_graph_components,
                ["prm_coal_mine_data", "prm_coal_mines_merged_with_railways"],
                ["coal_mines_nodes_dataframe", "coal_mine_railway_edge_dataframe"],
                tags=tags + ["coal_mine_graph"],
            ),
            node(
                create_city_node_table,
                ["prm_cities_data", "parameters"],
                "cities_nodes_dataframe",
                tags=tags + ["cities_nodes"],
            ),
            node(
                create_city_pipeline_edges,
                "prm_cities_matched_with_pipelines_and_railways",
                "cities_pipelines_edge_dataframe",
                tags=tags + ["cities_pipelines_edges"],
            ),
            node(
                create_city_railway_edges,
                "prm_cities_matched_with_pipelines_and_railways",
                "cities_railways_edge_dataframe",
                tags=tags + ["cities_railways_edges"],
            ),
            node(
                create_city_port_edges,
                "prm_cities_matched_with_pipelines_and_railways",
                "cities_ports_edge_dataframe",
                tags=tags + ["cities_ports_edges"],
            ),
        ]
    )

