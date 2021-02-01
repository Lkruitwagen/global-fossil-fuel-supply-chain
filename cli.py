import click, yaml, os, logging

from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from ffsc.pipeline.pipeline import get_pipeline as get_base_pipeline
from ffsc.flow.flow_pipeline import get_pipeline as get_flow_pipeline
from ffsc.communities.community_pipeline import get_pipeline as get_community_pipeline
from ffsc.interdiction.interdiction_pipeline import get_pipeline as get_interdiction_pipeline
from ffsc.visualise.visualise_pipeline import get_pipeline as get_visualise_pipeline

catalog = yaml.load(open(os.path.join(os.getcwd(),'conf','base','catalog.yml'),'r'),Loader=yaml.SafeLoader)
kedro_catalog = DataCatalog.from_config(catalog)

logging.basicConfig(level=logging.INFO)
logger = logger.getLogger('CLI')

def run_pipeline_dict(pipelines, logger):
    runner = SequentialRunner()

    for kk,pipeline in pipelines.items():
        logger.info(f'Running pipeline {kk}')
        runner.run(pipeline, kedro_catalog)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--tags', default=None, help='Optionally specify any individual node tags you want to run in a comma-separated list.')
def network_assembly(tags):
    """
    Assemble the basic network from asset-level data. See ffsc.pipeline.pipeline.py for detailed tags. 
    
    \b
    AVAILABLE TOP-LEVEL TAGS:
    -------------------------
    --preprocess : Preprocessing and homogenisation of all raw asset data.
    --prep       : Geospatial preparation operations on all data.
    --sjoin      : Spatial join operations matching linear and point assets.
    --flmile     : First- and last-mile matching operations to gapfill missing data.
    --explode    : Geospatial post-processing of joining and matching.
    --simplify   : Simplification operations to reduce the number of nodes.
    """
    
    if not tags:
        pipelines = {
            'preprocess': get_base_pipeline('preprocess'),
            'prep': get_base_pipeline('prep'),
            'sjoin': get_base_pipeline('sjoin'),
            'flmile': get_base_pipeline('flmile'),
            'explode': get_base_pipeline('explode'),
            'simplify': get_base_pipeline('simplify'),
        }
    else:
        pipelines = {kk:get_base_pipeline(kk) for kk in tags.split(',')}

    run_pipeline_dict(pipelines,logger)
    logger.info('DONE NETWORK ASSEMBLY')
    
    
@cli.command()
@click.option('--tags', default=None, help='Optionally specify any individual node tags you want to run')
def solve_flow(tags):
    """
    Assemble the basic network from asset-level data. See ffsc.flow.flow_pipeline.py, ffsc.communities.community_pipeline.py, and ffsc.interdiction.interdiction_pipeline.py for detailed tags. 
    
    \b
    AVAILABLE TOP-LEVEL TAGS:
    -------------------------
    --flow_edges           : Prepare network edges dataframe.
    --flow_nodes           : Prepare network nodes dataframe.
    --flow_nx              : Test network connectivity and prepared for flow calculations.
    --community-prep       : Prepare to add communities to network.
    --community-run        : Run community detection algorithm.
    --community-post-nodes : Post-process community detection onto node dataframe.
    --community-post-edges : Post-process community detection onto edge dataframe.
    --dijkstra-pickle      : Pickle edges in preparation for dijkstra mincost path.
    --dijkstra-paths       : Run async dijkstra mincost path.
    --dijkstra-adj         : Post-process dijkstra to mincost adjacency matrix.
    --dijkstra-flow        : Solve flow using iterative cost-scaling.
    """
    
    if not tags:
        pipelines = {
            'flow_edges'           : get_flow_pipeline('preprocess'),
            'flow_nodes'           : get_flow_pipeline('prep'),
            'flow_nx'              : get_flow_pipeline('sjoin'),
            'community-prep'       : get_community_pipeline('community-prep'),
            'community-run'        : get_community_pipeline('community-run'),
            'community-post-nodes' : get_community_pipeline('community-post-nodes'),
            'community-post-edges' : get_community_pipeline('community-post-edges'),
            'dijkstra-pickle'      : get_interdiction_pipeline('dijkstra-flow'),
            'dijkstra-paths'       : get_interdiction_pipeline('dijkstra-paths'),
            'dijkstra-adj'         : get_interdiction_pipeline('dijkstra-adj'),
            'dijkstra-flow'        : get_interdiction_pipeline('dijkstra-flow'),
        }
    else:
        pipelines = {kk:get_base_pipeline(kk) for kk in tags.split(',')}

    run_pipeline_dict(pipelines,logger)
    logger.info('DONE SOLVING FLOW')
        
    
    
@cli.command()
@click.option('--tags', default=None, help='Optionally specify any individual node tags you want to run')
def shock_analysis(tags):
    """
    Prepare demand and supply shock analysis. See ffsc.interdiction.interdiction_pipeline.py for detailed tags. 
    
    \b
    AVAILABLE TOP-LEVEL TAGS:
    -------------------------
    --sds_counterfactual       : Prepare Sustainable Development Scenario demand shock analysis.
    --supply-interdiction      : Prepare supply interdiction shock analysis.
    --post-supply-interdiction : Post-process supply interdiction shock analysis.
    """
    
    if not tags:
        pipelines = {
            'sds_counterfactual'       : get_interdiction_pipeline('counterfactual'),
            'supply-interdiction'      : get_interdiction_pipeline('supply-interdiction'),
            'post-supply-interdiction' : get_interdiction_pipeline('post-supply-interdiction'),
        }
    else:
        pipelines = {kk:get_base_pipeline(kk) for kk in tags.split(',')}

    run_pipeline_dict(pipelines,logger)
    logger.info('DONE SHOCK ANALYSIS')
    

@cli.command()
@click.option('--tags', default=None, help='Optionally specify any individual node tags you want to run')
def visualisation(tags):
    """
    Prepare visualisation of assets, flow, and demand shock counterfactual. See ffsc.visualisation.visualise_pipeline.py for detailed tags. 
    
    \b
    AVAILABLE TOP-LEVEL TAGS:
    -------------------------
    --visualise-assets      : Visualise all assets.
    --visualise-iso2        : Add iso2 country codes to dataframes.
    --visualise-trade-prep  : Prepare trade dataframes for comparison.
    --visualise-trade       : Visualise actual trade and production vs simulated.
    --visualise-flow        : Visualise energy flow.
    --compare-flow          : Compare energy flow to SDS demand shock energy flow.
    """
    
    if not tags:
        pipelines = {
            'visualise-assets'     : get_visualisation_pipeline('visualise-assets'),
            'visualise-iso2'       : get_visualisation_pipeline('visualise-iso2'),
            'visualise-trade-prep' : get_visualisation_pipeline('visualise-trade-prep'),
            'visualise-flow'       : get_visualisation_pipeline('visualise-flow'),
            'compare-flow'         : get_visualisation_pipeline('compare-flow')
        }
    else:
        pipelines = {kk:get_base_pipeline(kk) for kk in tags.split(',')}

    run_pipeline_dict(pipelines,logger)
    logger.info('DONE VISUALISATION')