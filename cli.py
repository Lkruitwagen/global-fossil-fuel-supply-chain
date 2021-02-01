import yaml, os

from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from ffsc.pipeline.pipeline import get_pipeline as get_base_pipeline
from ffsc.flow.flow_pipeline import get_pipeline as get_flow_pipeline
from ffsc.communities.community_pipeline import get_pipeline as get_community_pipeline

catalog = yaml.load(open(os.path.join(os.getcwd(),'conf','base','catalog.yml'),'r'),Loader=yaml.SafeLoader)


kedro_catalog = DataCatalog.from_config(catalog)#

#pipeline=get_base_pipeline('explode_edges_railways') # ran sjoin, now run flmile
#pipeline = get_base_pipeline('simplify')
pipeline = get_flow_pipeline("flow-solve_coal") # flow-nodes, flow-solve
#pipeline = get_community_pipeline("community-run")
#pipeline = get_base_pipeline("simpify_edges_shippingroutes") # flow-nodes, flow-solve
print (pipeline)
#pipelines = [
#    get_base_pipeline('simplify'),
#    get_flow_pipeline('flow-edges'),
#    get_flow_pipeline('flow-nodes'),
#    get_flow_pipeline('flow-solve')
#]

runner = SequentialRunner()
#
runner.run(pipeline,kedro_catalog)

#for pp in pipelines:
#    runner.run(pp, kedro_catalog)

"""
pipelines = {
    'A': get_base_pipeline('preprocess'),
    'B': get_base_pipeline('prep'),
    'C': get_base_pipeline('sjoin'),
    'D': get_base_pipeline('flmile'),
    'E': get_base_pipeline('explode'),
    'F': get_base_pipeline('simplify'),
}


runner = SequentialRunner()

for kk,pipeline in pipelines.items():
    runner.run(pipeline, kedro_catalog)
"""