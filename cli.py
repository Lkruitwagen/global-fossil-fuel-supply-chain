import yaml, os

from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from ffsc.pipeline.pipeline import get_pipeline as get_base_pipeline
from ffsc.flow.flow_pipeline import get_pipeline as get_flow_pipeline
from ffsc.communities.community_pipeline import get_pipeline as get_community_pipeline
from ffsc.interdiction.interdiction_pipeline import get_pipeline as get_interdiction_pipeline
from ffsc.visualise.visualise_pipeline import get_pipeline as get_visualise_pipeline

catalog = yaml.load(open(os.path.join(os.getcwd(),'conf','base','catalog.yml'),'r'),Loader=yaml.SafeLoader)


kedro_catalog = DataCatalog.from_config(catalog)

### Do missed
#pipeline = get_base_pipeline('simplify_missinglinks_cities')
#runner=SequentialRunner()
#runner.run(pipeline, kedro_catalog)
#exit()


### Do Vis
pipeline = get_visualise_pipeline(['compare-flow_oil'])
runner = SequentialRunner()
runner.run(pipeline, kedro_catalog)
exit()


"""
### Do an interdiction
pipeline = get_interdiction_pipeline(["counterfactual_oil"])#["dijkstra-genetic_gas"])#'dijkstra-paths_oil'])#])#'dijkstra-pickle_oil']) # flow-nodes, flow-solve
runner = SequentialRunner()
runner.run(pipeline,kedro_catalog)
exit()
"""

"""
### Do the communities
community_pipeline = [
    get_community_pipeline('community-prep'),
    get_community_pipeline('community-run'),
    get_community_pipeline('community-post-nodes'),
    get_community_pipeline('community-post-edges'),
    #get_community_pipeline("community-post-edges_oil")
]
runner = SequentialRunner()
for pp in community_pipeline:
    runner.run(pp, kedro_catalog)
"""

"""
### Do the flows
pipelines = [
#    get_base_pipeline('simplify'),
    get_flow_pipeline('flow-edges'),
    get_flow_pipeline('flow-nodes'),
    get_flow_pipeline('flow-prepnx')
]
runner = SequentialRunner()
for pp in pipelines:
    runner.run(pp, kedro_catalog)
"""

#pipelines = {
    #'A': get_base_pipeline('preprocess'),
    #'B': get_base_pipeline('prep'),
    #'C': get_base_pipeline('sjoin'),
#    'D': get_base_pipeline('flmile'),
#    'E': get_base_pipeline('explode'),
#    'F': get_base_pipeline('simplify'),
#}


#runner = SequentialRunner()

#for kk,pipeline in pipelines.items():
#    runner.run(pipeline, kedro_catalog)
