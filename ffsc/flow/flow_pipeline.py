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


from ffsc.flow.make_network import (
    make_coal_network,
    make_oil_network,
    make_gas_network
)
from ffsc.flow.prep_flow import (
    prep_oilwells,
    prep_coalmines,
    prep_oilfields,
    prep_cities
)
from ffsc.flow.network_flow import (
    prep_coal_nx,
)

def get_pipeline(tag=None):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """


    data_science_pipeline = (
        flow_prep_edges()
        + flow_prep_nodes()
        + flow_solve_mincost()
        #+ prep_pipeline()
    )
    
    if tag:
        if type(tag)==str:
            return Pipeline([n for n in data_science_pipeline.nodes if tag in n.tags])
        elif type(tag)==list:
            return Pipeline([n for n in data_science_pipeline.nodes if np.isin(n.tags,tag).any()])
        
    else:
        return data_science_pipeline


def flow_prep_edges(**kwargs):
    """The make_networks pipeline makes the base network edges for coal, oil and gas networks in parquet format."""
    tags = ["flow-edges"]
    
    INPUTS_COAL = [
        'simplify_cities_data', 
        'simplify_powerstations_data',
        'simplify_coalmines_data',
        'simplify_edges_cities',
        'simplify_edges_powerstations',
        'simplify_edges_coalmines',
        'simplify_edges_other_railways',
        'simplify_edges_railways_other',
        'simplify_edges_shippingroutes_other',
        'simplify_edges_railways_railways',
        'simplify_edges_shippingroutes_shippingroutes', 
        'flow_parameters'
    ]
    
    INPUTS_OILGAS = [
        'simplify_cities_data', 
        'simplify_powerstations_data',
        'simplify_oilfields_data',
        'simplify_oilwells_data',
        'simplify_edges_cities',
        'simplify_edges_powerstations',
        'simplify_edges_oilfields',
        'simplify_edges_oilwells',
        'simplify_edges_other_pipelines',
        'simplify_edges_pipelines_other',
        'simplify_edges_pipelines_pipelines',
        'simplify_edges_shippingroutes_other',
        'simplify_edges_shippingroutes_shippingroutes', 
        'flow_parameters'
    ]
    
    nodes = [
            node(
                make_coal_network,
                INPUTS_COAL,
                "flow_coal_edges",
                tags=tags + ["flow-make_coal"],
            ),
            node(
                make_oil_network,
                INPUTS_OILGAS,
                "flow_oil_edges",
                tags=tags + ["flow-make_oil"]
            ),
            node(
                make_gas_network,
                INPUTS_OILGAS,
                "flow_gas_edges",
                tags=tags + ["flow-make_gas"]
            ),
        
        ]
        
    return Pipeline(nodes)

    
    
def flow_prep_nodes(**kwargs):
    """The make_networks pipeline makes the base network edges for coal, oil and gas networks in parquet format."""
    tags = ["flow-nodes"]
    
    nodes = [
            node(
                prep_coalmines,
                ["raw_coalmines_data","iso2","ne"],
                "flow_coalmines_data",
                tags=tags + ["flow-prep_coalmines"],
            ),
            node(
                prep_oilfields,
                ["raw_oilfields_data","iso2"],
                "flow_oilfields_data",
                tags=tags + ["flow-prep_oilfields"]
            ),
            node(
                prep_oilwells,
                ["raw_oilwells_data","iso2"],
                "flow_oilwells_data",
                tags=tags + ["flow-prep_oilwells"]
            ),
            node(
                prep_cities,
                ["raw_cities_euclidean_data"],
                "flow_cities_data",
                tags=tags + ["flow-prep_cities"]
            )
        ]
        
    return Pipeline(nodes)


def flow_solve_mincost(**kwargs):
    
    tags = ["flow-solve"]
    
    nodes = [
        node(
            prep_coal_nx,
            ["flow_coal_edges", "flow_cities_data", "simplify_powerstations_data", "flow_coalmines_data", "global_energy_production", "ne", "flow_parameters"],
            ["flow_coal_nx_edges","flow_coal_nx_nodes"],
            tags = tags + ["flow-solve_coal"]
        )
    ]
    
    return Pipeline(nodes)
