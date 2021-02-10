"""Construction of the master pipeline.
"""

from typing import Dict
from kedro.pipeline import Pipeline, node


from ffsc.communities.community_methods import (
    format_textfiles,
    run_communities,
    post_community_nodes,
    post_community_edges
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
        + community_post_nodes_pl()
        + community_post_edges_pl()
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

    
    
def community_post_nodes_pl(**kwargs):
    """post-process the DirectedLouvain results."""
    tags = ["community-post-nodes"]
    
    nodes = [
            node(
                post_community_nodes,
                ["flow_coal_nx_nodes","flow_oil_nx_nodes","flow_gas_nx_nodes"],
                ["community_coal_nodes_interim","community_oil_nodes_interim","community_gas_nodes_interim"],
                tags=tags,
            ),
        ]
        
    return Pipeline(nodes)

    
def community_post_edges_pl(**kwargs):
    """post-process the DirectedLouvain results."""
    tags = ["community-post-edges"]
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
    
    nodes = [
            node(
                post_community_edges,
                ["vis_parameters","flow_coal_nx_edges"] + PT_ASSETS + ['community_coal_nodes_interim'],
                ["community_coal_nodes","community_coal_edges","communities_coal"],
                tags=tags + ["community-post-edges_coal"],
            ),
            node(
                post_community_edges,
                ["vis_parameters","flow_gas_nx_edges"] + PT_ASSETS + ['community_gas_nodes_interim'],
                ["community_gas_nodes","community_gas_edges","communities_gas"],
                tags=tags + ["community-post-edges_gas"],
            ),
            node(
                post_community_edges,
                ["vis_parameters","flow_oil_nx_edges"] + PT_ASSETS + ['community_oil_nodes_interim'],
                ["community_oil_nodes","community_oil_edges","communities_oil"],
                tags=tags + ["community-post-edges_oil"],
            ),
        ]
        
    return Pipeline(nodes)

