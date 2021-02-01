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
    )
    
    if tag:
        if type(tag)==str:
            return Pipeline([n for n in data_science_pipeline.nodes if tag in n.tags])
        elif type(tag)==list:
            return Pipeline([n for n in data_science_pipeline.nodes if len(n.tags - set(tag)) < len(n.tags)])
        
    else:
        return data_science_pipeline


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
            interdiction_community_coal,
            ["flow_parameters"],
            [],
            tags = tags + ["interdiction-community_coal"]
        ),
        node(
            interdiction_community_oil,
            ["flow_parameters"],
            [],
            tags = tags + ["interdiction-community_oil"]
        ),
        node(
            interdiction_community_gas,
            ["flow_parameters"],
            [],
            tags = tags + ["interdiction-community_gas"]
        ),
    ]
    
    return Pipeline(nodes)
