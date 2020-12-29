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

from ffsc.interdiction.interdiction_methods import (
    interdiction_baseline_call,
    interdiction_baseline_parse,
    interdiction_community_coal,
    interdiction_community_oil,
    interdiction_community_gas,
    interdiction_community,
    interdiction_community_parse,
)
from ffsc.interdiction.dijkstra_methods import (
    dijkstra_prep_paths,
    dijkstra_parse_paths,
    dijkstra_shortest_allpairs,
    dijkstra_filter_reachable,
)

def get_pipeline(tag=None):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """


    data_science_pipeline = (
        interdiction_baseline_call_pl()
        + interdiction_baseline_parse_pl()
        + interdiction_community_pl()
        + interdiction_community_parse_pl()
        + dijkstra_prep_paths_pl()
        + dijkstra_parse_paths_pl()
        + dijkstra_reachable_pl()
        + dijkstra_shortest_paths_pl()
    )
    
    if tag:
        if type(tag)==str:
            return Pipeline([n for n in data_science_pipeline.nodes if tag in n.tags])
        elif type(tag)==list:
            return Pipeline([n for n in data_science_pipeline.nodes if len(n.tags - set(tag)) < len(n.tags)])
        
    else:
        return data_science_pipeline
    
################################
### new interdiction methods ###
################################

def dijkstra_prep_paths_pl(**kwargs):
    tags = ["dijkstra-prep"]
    
    nodes = [
        node(
            dijkstra_prep_paths,
            ['community_coal_nodes', 'flow_coal_nx_edges', 'flow_parameters'],
            [],
            tags = tags + ['dijkstra-prep_coal']
        )
    ]
    
    return Pipeline(nodes)

def dijkstra_parse_paths_pl(**kwargs):
    tags = ["dijkstra-parse"]
    
    nodes = [
        node(
            dijkstra_parse_paths,
            ['community_coal_nodes', 'flow_parameters'],
            'coal_nodeflags',
            tags = tags + ['dijkstra-parse_coal']
        )
    ]
    
    return Pipeline(nodes)

def dijkstra_reachable_pl(**kwargs):
    tags = ["dijkstra-reachable"]
    
    nodes = [
        node(
            dijkstra_filter_reachable,
            ['community_coal_nodes', 'flow_parameters'],
            'coal_reachable',
            tags = tags + ['dijkstra-reachable_coal']
        )
    ]
    
    return Pipeline(nodes)

def dijkstra_shortest_paths_pl(**kwargs):
    tags = ["dijkstra-shortest"]
    
    nodes = [
        node(
            dijkstra_shortest_allpairs,
            ['community_coal_nodes', 'flow_coal_nx_edges', 'coal_nodeflags', 'coal_reachable', 'flow_parameters'], 
            'coal_dijkstra_pairs',
            tags = tags + ['dijkstra-shortest_coal']
        )
    ]
    
    return Pipeline(nodes)


def interdiction_baseline_call_pl(**kwargs):
    
    tags = ["interdiction-baseline"]
    
    nodes = [
        node(
            interdiction_baseline_call,
            ["flow_parameters"],
            [],
            tags = tags
        ),
    ]
    
    return Pipeline(nodes)

def interdiction_baseline_parse_pl(**kwargs):
    
    tags = ["interdiction-baseline-parse"]
    
    nodes = [
        node(
            interdiction_baseline_parse,
            ["flow_coal_nx_nodes","flow_oil_nx_nodes","flow_gas_nx_nodes"],
            ["flow_baseline_coal","flow_baseline_oil","flow_baseline_gas", "flow_baseline_costs"],
            tags = tags
        ),
    ]
    
    return Pipeline(nodes)

def interdiction_community_pl(**kwargs):
    
    tags = ["interdiction-community"]
    
    nodes = [
        node(
            interdiction_community,
            ["community_coal_nodes", "community_coal_edges", "communities_coal", "flow_baseline_coal", "flow_parameters"],
            [],
            tags = tags + ["interdiction-community_coal"]
        ),
        node(
            interdiction_community,
            ["community_oil_nodes", "community_oil_edges", "communities_oil", "flow_baseline_oil", "flow_parameters"],
            [],
            tags = tags + ["interdiction-community_oil"]
        ),
        node(
            interdiction_community,
            ["community_gas_nodes", "community_gas_edges", "communities_gas", "flow_baseline_gas", "flow_parameters"],
            [],
            tags = tags + ["interdiction-community_gas"]
        ),
    ]
    
    return Pipeline(nodes)

def interdiction_community_parse_pl(**kwargs):
    
    tags = ["interdiction-community-parse"]
    
    nodes = [
        node(
            interdiction_community_parse,
            ["community_coal_nodes"],
            [],
            tags = tags + ["interdiction-community-parse_coal"]
        ),
        node(
            interdiction_community_parse,
            ["community_oil_nodes"],
            [],
            tags = tags + ["interdiction-community-parse_oil"]
        ),
        node(
            interdiction_community_parse,
            ["community_gas_nodes"],
            [],
            tags = tags + ["interdiction-community-parse_gas"]
        ),
    ]
    
    return Pipeline(nodes)
