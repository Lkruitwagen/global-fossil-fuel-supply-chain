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


from ffsc.communities.community_methods import (
    format_textfiles,
    run_communities,
    post_communities
)

def get_pipeline(tag=None):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """


    data_science_pipeline = (
        community_prep()
        + community_run()
        + community_post()
    )
    
    if tag:
        if type(tag)==str:
            return Pipeline([n for n in data_science_pipeline.nodes if tag in n.tags])
        elif type(tag)==list:
            return Pipeline([n for n in data_science_pipeline.nodes if np.isin(n.tags,tag).any()])
        
    else:
        return data_science_pipeline


def community_prep(**kwargs):
    """community_prep prepares the text files."""
    
    tags = ["community-prep"]
    
    
    nodes = [
            node(
                format_textfiles,
                ["flow_coal_nx_nodes","flow_coal_nx_edges"],
                [],
                tags=tags + ["community-prep_coal"],
            ),
            node(
                format_textfiles,
                ["flow_oil_nx_nodes","flow_oil_nx_edges"],
                [],
                tags=tags + ["community-prep_oil"]
            ),
            node(
                format_textfiles,
                ["flow_gas_nx_nodes","flow_gas_nx_edges"],
                [],
                tags=tags + ["community-prep_gas"]
            ),
        
        ]
        
    return Pipeline(nodes)


def community_run(**kwargs):
    """community_run prepares the text files."""
    
    tags = ["community-run"]
    
    
    nodes = [
            node(
                run_communities,
                ['dummy'],
                [],
                tags=tags,
            ),        
        ]
        
    return Pipeline(nodes)

    
    
def community_post(**kwargs):
    """post-process the DirectedLouvain results."""
    tags = ["community-post"]
    
    nodes = [
            node(
                post_communities,
                ["flow_coal_nx_nodes","flow_oil_nx_nodes","flow_gas_nx_nodes"],
                ["communities_coal_nx_nodes","communities_oil_nx_nodes","communities_gas_nx_nodes"],
                tags=tags,
            ),
        ]
        
    return Pipeline(nodes)

