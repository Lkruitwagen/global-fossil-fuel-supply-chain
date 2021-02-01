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

from ffsc.visualise.visualise import (
    visualise_communities_detail,
    visualise_flow,
    visualise_assets_coal,
    visualise_assets_oil,
    visualise_assets_gas,
    visualise_assets_simplified_coal,
    visualise_assets_simplified_oil,
    visualise_assets_simplified_gas,
    visualise_communities_blobs,
    visualise_trade,
    node_iso2,
    flow2iso2adj,
)

PT_ASSETS = [
    'simplify_refineries_data',
    'simplify_oilfields_data',
    'simplify_oilwells_data',
    'simplify_coalmines_data', 
    'simplify_lngterminals_data', 
    'simplify_ports_data', 
    'simplify_cities_data', 
    'simplify_powerstations_data',
]
LIN_ASSETS = [
    'XXX_edges_pipelines_pipelines',
    'XXX_edges_railways_railways',
    'XXX_edges_shippingroutes_shippingroutes'
]
MISSING_ASSETS = [
    'flow_XXX_missing_cities',
    'flow_XXX_missing_powerstations'
]


def get_pipeline(tag=None):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """


    data_science_pipeline = (
        vis_assets()
        + vis_communities_blobs_pl()
        + vis_individual_communities()
        + vis_trade_pl()
        + node_iso2_pl()
        + vis_trade_prep_pl()
        + vis_flow_pl()
    )
    
    if tag:
        if type(tag)==str:
            return Pipeline([n for n in data_science_pipeline.nodes if tag in n.tags])
        elif type(tag)==list:
            return Pipeline([n for n in data_science_pipeline.nodes if len(n.tags - set(tag)) < len(n.tags)])
        
    else:
        return data_science_pipeline



def vis_assets(**kwargs):
    
    tags = ["visualise-assets"]
    
    
    
    nodes = [
        node(
            visualise_assets_coal,
            ["vis_parameters","flow_coal_nx_edges"]+PT_ASSETS+[el.replace('XXX','coal') for el in MISSING_ASSETS]+[el.replace('XXX','explode') for el in LIN_ASSETS]+['ne'],
            [],
            tags = tags + ["visualise-assets_coal"]
        ),
        node(
            visualise_assets_oil,
            ["vis_parameters","flow_oil_nx_edges"]+PT_ASSETS+[el.replace('XXX','oil') for el in MISSING_ASSETS]+[el.replace('XXX','explode') for el in LIN_ASSETS]+['ne'],
            [],
            tags = tags + ["visualise-assets_oil"]
        ),
        node(
            visualise_assets_gas,
            ["vis_parameters","flow_gas_nx_edges"]+PT_ASSETS+[el.replace('XXX','gas') for el in MISSING_ASSETS]+[el.replace('XXX','explode') for el in LIN_ASSETS]+['ne'],
            [],
            tags = tags + ["visualise-assets_gas"]
        ),
    ]
    
    return Pipeline(nodes)

def vis_assets_simplified(**kwargs):
    
    tags = ["visualise-assets-simplified"]
    
    nodes = [
        node(
            visualise_assets_simplified_coal,
            ["vis_parameters","flow_coal_nx_edges"]+PT_ASSETS+[el.replace('XXX','coal') for el in MISSING_ASSETS]+[el.replace('XXX','simplify') for el in LIN_ASSETS]+['ne'],
            [],
            tags = tags + ["visualise-assets-simplified_coal"]
        ),
        node(
            visualise_assets_simplified_oil,
            ["vis_parameters","flow_oil_nx_edges"]+PT_ASSETS+[el.replace('XXX','oil') for el in MISSING_ASSETS]+[el.replace('XXX','simplify') for el in LIN_ASSETS]+['ne'],
            [],
            tags = tags + ["visualise-assets-simplied_oil"]
        ),
        node(
            visualise_assets_simplified_gas,
            ["vis_parameters","flow_gas_nx_edges"]+PT_ASSETS+[el.replace('XXX','gas') for el in MISSING_ASSETS]+[el.replace('XXX','simplify') for el in LIN_ASSETS]+['ne'],
            [],
            tags = tags + ["visualise-assets-simplified_gas"]
        ),
    ]
    
    return Pipeline(nodes)

def vis_communities_blobs_pl(**kwargs):
    tags = ["visualise-communities-blobs"]
    
    nodes = [
        node(
            visualise_communities_blobs,
            ["communities_coal", "ne","vis_parameters"],
            [],
            tags = tags + ["visualise-community-blobs_coal"]
        ),
        node(
            visualise_communities_blobs,
            ["communities_oil", "ne","vis_parameters"],
            [],
            tags = tags + ["visualise-community-blobs_oil"]
        ),
        node(
            visualise_communities_blobs,
            ["communities_gas", "ne","vis_parameters"],
            [],
            tags = tags + ["visualise-community-blobs_gas"]
        ),
        
    ]
    
    return Pipeline(nodes)
def vis_individual_communities(**kwargs):
    tags = ["visualise-communities-details"]
    
    nodes = [
        node(
            visualise_communities_detail,
            ["vis_parameters"]+[el.replace('XXX','coal') for el in ["community_XXX_nodes", "community_XXX_edges", "flow_baseline_XXX", "communities_XXX"]]+["ne"],
            [],
            tags = tags + ["visualise-community-details_coal"]
        ),
        node(
            visualise_communities_detail,
            ["vis_parameters"]+[el.replace('XXX','oil') for el in ["community_XXX_nodes", "community_XXX_edges", "flow_baseline_XXX", "communities_XXX"]]+["ne"],
            [],
            tags = tags + ["visualise-community-details_oil"]
        ),
        node(
            visualise_communities_detail,
            ["vis_parameters"]+[el.replace('XXX','gas') for el in ["community_XXX_nodes", "community_XXX_edges","flow_baseline_XXX", "communities_XXX"]]+["ne"],
            [],
            tags = tags + ["visualise-community-details_gas"]
        ),
        
    ]
    
    return Pipeline(nodes)

def vis_trade_prep_pl(**kwargs):
    tags = ["visualise-trade-prep"]
    
    nodes = [
        node(
            flow2iso2adj,
            ["iso2", "flow_baseline_coal", "community_coal_iso2"],
            "flow_coal_iso2",
            tags=tags + ["visualise-trade-prep_coal"]
        ),
        node(
            flow2iso2adj,
            ["iso2", "flow_baseline_oil", "community_oil_iso2"],
            "flow_oil_iso2",
            tags=tags + ["visualise-trade-prep_oil"]
        ),
        node(
            flow2iso2adj,
            ["iso2", "flow_baseline_gas", "community_gas_iso2"],
            "flow_gas_iso2",
            tags=tags + ["visualise-trade-prep_gas"]
        ),
        
    ]
    
    return Pipeline(nodes)

def vis_trade_pl(**kwargs):
    tags = ["visualise-trade"]
    
    nodes = [
        node(
            visualise_trade,
            ["iso2", "ne", "global_energy_production", "coal_trade", "flow_coal_iso2"],
            [],
            tags=tags + ["visualise-trade_coal"]
        ),
        node(
            visualise_trade,
            ["iso2", "ne", "global_energy_production", "oil_trade", "flow_oil_iso2"],
            [],
            tags=tags + ["visualise-trade_oil"]
        ),
        node(
            visualise_trade,
            ["iso2", "ne", "global_energy_production", "gas_trade", "flow_gas_iso2"],
            [],
            tags=tags + ["visualise-trade_gas"]
        ),
    ]
    
    return Pipeline(nodes)

def node_iso2_pl(**kwargs):
    tags = ['visualise-iso2']
    
    nodes = [
        node(
            node_iso2,
            ["iso2","ne","community_coal_nodes","community_oil_nodes","community_gas_nodes","raw_oilfields_data","raw_oilwells_data"],
            ["community_coal_iso2","community_oil_iso2","community_gas_iso2"],
            tags = tags
        ),
    ]
    
    return Pipeline(nodes)

def vis_flow_pl(**kwargs):
    tags = ['visualise-flow']
    
    nodes = [
        node(
            visualise_flow,
            ["vis_parameters", "ne", "flow_baseline_coal", "community_coal_edges", "community_coal_nodes"],
            [],
            tags = tags+['visualise-flow_coal']
        ),
        node(
            visualise_flow,
            ["vis_parameters", "ne", "flow_baseline_oil", "community_oil_edges", "community_oil_nodes"],
            [],
            tags = tags+['visualise-flow_oil']
        ),
        node(
            visualise_flow,
            ["vis_parameters", "ne", "flow_baseline_gas", "community_gas_edges", "community_gas_nodes"],
            [],
            tags = tags+['visualise-flow_gas']
        )
    ]
    
    return Pipeline(nodes)
