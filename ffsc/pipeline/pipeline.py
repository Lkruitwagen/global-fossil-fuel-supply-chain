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


from ffsc.pipeline.nodes.preprocess import (
    preprocess_shippingroutes,
    preprocess_ports,
    preprocess_pipelines,
    preprocess_coalmines,
    preprocess_oilfields,
    preprocess_lngterminals,
    preprocess_powerstations,
    preprocess_railways,
    preprocess_refineries,
    preprocess_oilwells,
    preprocess_cities_base,
)
from ffsc.pipeline.nodes.prep import (
    selfjoin_and_dissolve,
    buffer_and_split
)
from ffsc.pipeline.nodes.utils import (
    null_forward
)
from ffsc.pipeline.nodes.spatialjoin import (
    spatialjoin
)
from ffsc.pipeline.nodes.firstlastmile import (
    firstmile_edge,
    powerstations_lastmile,
    cities_delauney
)
from ffsc.pipeline.nodes.explode import (
    explode_edges_railways,
    explode_edges_shippingroutes,
    explode_edges_pipelines,
    drop_linear
)
from ffsc.pipeline.nodes.simplify import (
    simplify_edges,
)
from ffsc.pipeline.nodes.internationaldateline import (
    connect_IDL    
)

ALL_SECTORS = ['shippingroutes','pipelines','railways','refineries','oilfields','oilwells','coalmines','lngterminals','ports','cities','powerstations']

SJOIN_PAIRS = [
        ('shippingroutes','lngterminals'),
        ('shippingroutes','ports'),
        ('pipelines','refineries'),
        ('pipelines','oilfields'),
        ('pipelines','oilwells'),
        ('pipelines','lngterminals'),
        ('pipelines','ports'),
        ('pipelines','cities'),
        ('pipelines','powerstations'),
        ('railways','coalmines'),
        ('railways','ports'),
        ('railways','cities'),
        ('railways','powerstations'),
    ]

def get_pipeline(tag=None):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """


    data_science_pipeline = (
        preprocess_pipeline()
        + prep_pipeline()
        + sjoin_pipeline()
        + firstlastmile_pipeline()
        + explode_pipeline()
        + simplify_pipeline()
    )
    
    if tag:
        if type(tag)==str:
            return Pipeline([n for n in data_science_pipeline.nodes if tag in n.tags])
        elif type(tag)==list:
            return Pipeline([n for n in data_science_pipeline.nodes if np.isin(n.tags,tag).any()])
        
    else:
        return data_science_pipeline


def preprocess_pipeline(**kwargs):
    """The pre-process pipeline generically imports the geojson and other files and writes them to a generic parquet format."""
    tags = ["preprocess"]
    return Pipeline(
        [
            node(
                preprocess_shippingroutes,
                "raw_shippingroutes_data",
                "prm_shippingroutes_data",
                tags=tags + ["preprocess_shippingroutes"],
            ),
            node(
                preprocess_ports,
                "raw_ports_data",
                "prm_ports_data",
                tags=tags + ["preprocess_ports"],
            ),
            node(
                preprocess_pipelines,
                "raw_pipelines_data",
                "prm_pipelines_data",
                tags=tags + ["preprocess_pipelines"],
            ),

            node(
                preprocess_coalmines,
                "raw_coalmines_data",
                "prm_coalmines_data",
                tags=tags + ["preprocess_coalmines"],
            ),
            node(
                preprocess_oilfields,
                "raw_oilfields_data",
                "prm_oilfields_data",
                tags=tags + ["preprocess_oilfields"],
            ),
            node(
                preprocess_lngterminals,
                "raw_lngterminals_data",
                "prm_lngterminals_data",
                tags=tags + ["preprocess_lngterminals"],
            ),
            node(
                preprocess_powerstations,
                "raw_powerstations_data",
                "prm_powerstations_data",
                tags=tags + ["preprocess_powerstations"],
            ),
            node(
                preprocess_railways,
                "raw_railways_data",
                "prm_railways_data",
                tags=tags + ["preprocess_railways"],
            ),
            node(
                preprocess_refineries,
                ["raw_refineries_data","raw_processingplants_data"],
                "prm_refineries_data",
                tags=tags + ["preprocess_refineries"],
            ),
            node(
                preprocess_oilwells,
                "raw_oilwells_data",
                "prm_oilwells_data",
                tags=tags + ["preprocess_wellpads"],
            ),
            node(
                preprocess_cities_base,
                ["raw_cities_energy_data"],
                "prm_cities_data",
                tags=tags + ["preprocess_cities"],
            ),
        ]
    )


def prep_pipeline(**kwargs):
    """The prep pipeline performs spatial operations on the data in preparation for spatial joining, accelerated by multiprocessing."""
    tags = ["prep"]
    
    sjd_nodes = [node(selfjoin_and_dissolve, f'prm_{sector}_data',f'prp_{sector}_data',tags=tags+['sjd',f'sjd_{sector}']) 
                 for sector in ['railways','pipelines'] ]
    
    buffer_and_split_nodes = [node(buffer_and_split,[f'prm_{sector}_data','parameters'],f'prp_{sector}_data',tags=tags+['buffer',f'buffer_{sector}']) 
                              for sector in ['refineries','oilfields','oilwells','coalmines','lngterminals','ports','cities','powerstations']]
    
    null_nodes = [node(null_forward,f'prm_{sector}_data',f'prp_{sector}_data',tags=tags+['prep_null',f'prep_null_{sector}']) 
                  for sector in ['shippingroutes']]
    
    
    return Pipeline(sjd_nodes + buffer_and_split_nodes + null_nodes)


def sjoin_pipeline(**kwargs):
    """ The sjoin pipeline performs spatial joins between the datasets"""
    tags = ["sjoin"]
    
    sjoin_nodes = [node(spatialjoin, [f'prp_{sector1}_data',f'prp_{sector2}_data'],f'sjoin_edges_{sector1}_{sector2}',tags=tags+['sjoin_mp',f'sjoin_{sector1}_{sector2}'])
                  for sector1, sector2 in SJOIN_PAIRS]
    null_nodes = [node(null_forward,f'prp_{sector}_data',f'sjoin_{sector}_data', tags=tags+['sjoin_null',f'sjoin_null_{sector}'])
                 for sector in ALL_SECTORS]

    return Pipeline(sjoin_nodes + null_nodes)


def firstlastmile_pipeline(**kwargs):
    """The first and last mile pipeline attaches any unattached elements to ensure a fully-connected graph"""
    tags = ['flmile']
    
    firstmile_nodes = [
        node(
            firstmile_edge,
            ['sjoin_oilfields_data','sjoin_edges_pipelines_oilfields','sjoin_ports_data','sjoin_cities_data','sjoin_pipelines_data'],
            'flmile_edges_oilfields',
            tags=tags+['firstmile','firstmile_oilfields']
        ), # assets, existing_edges, closest port, city, [pipeline/railway]
        node(
            firstmile_edge,
            ['sjoin_oilwells_data','sjoin_edges_pipelines_oilwells','sjoin_ports_data','sjoin_cities_data','sjoin_pipelines_data'],
            'flmile_edges_oilwells',
            tags=tags+['firstmile','firstmile_oilwells']
        ), # assets, existing_edges, closest port, city, [pipeline/railway]
        node(
            firstmile_edge,
            ['sjoin_coalmines_data','sjoin_edges_railways_coalmines','sjoin_ports_data','sjoin_cities_data','sjoin_railways_data'],
            'flmile_edges_coalmines',
            tags=tags+['firstmile','firstmile_coalmines']
        ), # assets, existing_edges, closest port, city, [pipeline/railway]
    ] 
    lastmile_nodes = [
        node(
            powerstations_lastmile,
            ['sjoin_powerstations_data','sjoin_edges_pipelines_powerstations','sjoin_edges_railways_powerstations','sjoin_railways_data','sjoin_pipelines_data','sjoin_ports_data','sjoin_cities_data'], # powerstations, ps_edges_pipelines, ps_edges_railways, railways, pipelines, ports, cities
            'flmile_edges_powerstations',
            tags=tags+['lastmile','lastmile_powerstations']
        )
    ]
    lastmile_nodes += [
        node(
            cities_delauney,
            ['sjoin_cities_data','ne'],
            'flmile_edges_cities',
            tags = tags+['lastmile','lastmile_cities']
        )
    ]
    
    IDL_nodes = [
        node(
            connect_IDL,
            'sjoin_shippingroutes_data',
            'flmile_idl_edges',
            tags=tags+['flmile_idl']
        )
    ]
    
    null_nodes = [node(null_forward, f'sjoin_{sector}_data', f'flmile_{sector}_data',tags = tags+['flm_null',f'flm_null_{sector}'])
                 for sector in ALL_SECTORS]
    null_nodes += [node(null_forward, f'sjoin_edges_{sector1}_{sector2}', f'flmile_edges_{sector1}_{sector2}', tags=tags+['flm_null',f'flm_null_{sector1}_{sector2}'])
                  for sector1, sector2 in SJOIN_PAIRS]
    
    return Pipeline(firstmile_nodes + lastmile_nodes + IDL_nodes + null_nodes)
    

def explode_pipeline(**kwargs):
    """penultimate step explodes linear data into points and edges """
    tags=['explode']
    
    PIPELINE_PAIRS = [
        ('pipelines','refineries'),
        ('pipelines','oilfields'),
        ('pipelines','oilwells'),
        ('pipelines','lngterminals'),
        ('pipelines','ports'),
        ('pipelines','cities'),
        ('pipelines','powerstations')
    ]
    
    RAILWAY_PAIRS = [
        ('railways','coalmines'),
        ('railways','ports'),
        ('railways','cities'),
        ('railways','powerstations'),
    ]
    
    PIPELINE_EDGES = [f'flmile_edges_{sector1}_{sector2}' for sector1, sector2 in PIPELINE_PAIRS] + ['flmile_edges_powerstations','flmile_edges_oilfields','flmile_edges_oilwells']
    RAILWAY_EDGES = [f'flmile_edges_{sector1}_{sector2}' for sector1, sector2 in RAILWAY_PAIRS] + ['flmile_edges_powerstations','flmile_edges_coalmines']
    SHIPPINGROUTES_EDGES = [f'flmile_edges_{sector1}_{sector2}' for sector1,sector2 in [('shippingroutes','lngterminals'),('shippingroutes','ports')]]
    
    explode_nodes = [
        node(
            explode_edges_pipelines,
            ['flmile_pipelines_data']+PIPELINE_EDGES,
            ['explode_pipelines_data', 'explode_edges_pipelines_pipelines','explode_keepnodes_pipelines','explode_edges_pipelines_other', 'explode_edges_other_pipelines'],
            tags=tags+['explode_edges','explode_edges_pipelines']
        ),
        node(
            explode_edges_railways,
            ['flmile_railways_data']+RAILWAY_EDGES,
            ['explode_railways_data', 'explode_edges_railways_railways','explode_keepnodes_railways','explode_edges_railways_other', 'explode_edges_other_railways'],
            tags=tags+['explode_edges','explode_edges_railways']
        ),
        node(
            explode_edges_shippingroutes,
            ['flmile_shippingroutes_data']+SHIPPINGROUTES_EDGES + ['flmile_idl_edges'],
            ['explode_shippingroutes_data', 'explode_edges_shippingroutes_shippingroutes','explode_keepnodes_shippingroutes','explode_edges_shippingroutes_other'],
            tags=tags+['explode_edges','explode_edges_shippingroutes']
        ),
    ]
    
    drop_nodes = [
        node(drop_linear,
             f'flmile_edges_{sector}',
             f'explode_edges_{sector}',
             tags=tags+['drop_linear', f'drop_linear_{sector}']
        )
        for sector in ['powerstations','oilfields','oilwells','coalmines']
    ]
    
    null_nodes = [
        node(null_forward,
             f'flmile_{sector}_data',
             f'explode_{sector}_data',
             tags=tags+['explode_null',f'explode_null_{sector}']
            )
        for sector in ['oilfields','oilwells','refineries','coalmines','lngterminals','ports','cities','powerstations']
    ]
    
    null_nodes += [
        node(null_forward,
             f'flmile_edges_cities',
             f'explode_edges_cities',
             tags=tags+['explode_null',f'explode_null_cities']
        )
    ]
    

    
    return Pipeline(explode_nodes+drop_nodes+null_nodes)

    
def simplify_pipeline(**kwargs):
    """final step simplified linear data for network assembly """
    tags=['simplify']
    
    simplify_nodes = [
        node(
            simplify_edges,
            ['explode_pipelines_data', 'explode_edges_pipelines_pipelines', 'explode_keepnodes_pipelines'],
            ['simplify_pipelines_data', 'simplify_edges_pipelines_pipelines'],
            tags=tags+['simplify_edges','simpify_edges_pipelines']
        ),
        node(
            simplify_edges,
            ['explode_railways_data', 'explode_edges_railways_railways', 'explode_keepnodes_railways'],
            ['simplify_railways_data', 'simplify_edges_railways_railways'],
            tags=tags+['simplify_edges','simpify_edges_railways']
        ),
        node(
            simplify_edges,
            ['explode_shippingroutes_data', 'explode_edges_shippingroutes_shippingroutes', 'explode_keepnodes_shippingroutes'],
            ['simplify_shippingroutes_data', 'simplify_edges_shippingroutes_shippingroutes'],
            tags=tags+['simplify_edges','simpify_edges_shippingroutes']
        ),
    ]
    
    null_nodes = [
        node(null_forward,
             f'explode_{sector}_data',
             f'simplify_{sector}_data',
             tags=tags+['simplify_null',f'null_{sector}']
            )
        for sector in ['oilfields','oilwells','refineries','coalmines','lngterminals','ports','cities','powerstations']
    ]
    
    null_nodes += [
        node(null_forward,
             f'explode_edges_{sector}',
             f'simplify_edges_{sector}',
             tags=tags+['simplify_null',f'null_edges_{sector}']
            )
        for sector in ['oilfields','oilwells','coalmines','cities','powerstations']
    ]
    
    null_nodes += [
        node(null_forward,
             f'explode_edges_other_{sector}',
             f'simplify_edges_other_{sector}',
             tags=tags+['simplify_null',f'simplify_null_edges_{sector}']
            )
        for sector in ['pipelines','railways']
    ]
    
    null_nodes += [
        node(null_forward,
             f'explode_edges_{sector}_other',
             f'simplify_edges_{sector}_other',
             tags=tags+['simplify_null',f'simplify_null_edges_{sector}']
            )
        for sector in ['pipelines','railways','shippingroutes']
    ]
    
    return Pipeline(simplify_nodes+null_nodes)

