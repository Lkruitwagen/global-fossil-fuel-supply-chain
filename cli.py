import yaml, os

from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from ffsc.pipeline.pipeline import get_pipeline

catalog = yaml.load(open(os.path.join(os.getcwd(),'conf','base','catalog.yml'),'r'),Loader=yaml.SafeLoader)


kedro_catalog = DataCatalog.from_config(catalog)#

pipeline=get_pipeline('simpify_edges_pipelines') # ran sjoin, now run flmile

runner = SequentialRunner()

runner.run(pipeline, kedro_catalog)

"""
pipelines = {
    'A': get_pipeline('preprocess'),
    'B': get_pipeline('prep'),
    'C': get_pipeline('sjoin'),
    'D': get_pipeline('flmile'),
    'E': get_pipeline('explode'),
    'F': get_pipeline('simplify'),
}


runner = SequentialRunner()

for kk,pipeline in pipelines.items():
    runner.run(pipeline, kedro_catalog)
"""