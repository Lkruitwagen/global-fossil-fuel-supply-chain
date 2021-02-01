import yaml, glob, json, os, pickle, logging
import kedro
from kedro.io import DataCatalog
import pandas as pd
import numpy as np
import networkx as nx

catalog = yaml.load(open(os.path.join(os.getcwd(),'conf','base','catalog.yml'),'r'),Loader=yaml.SafeLoader)
kedro_catalog = DataCatalog.from_config(catalog)

logging.basicConfig(filename=os.path.join(os.getcwd(),'summary.txt'), level=logging.INFO)
logger = logging.getLogger('SUMMARY')

def simplify_stats():
    
    ### want number of nodes before/after simplify
    logger.info('summaryising simplification')
    for lin_type in ['pipelines','railways','shippingroutes']:
        exp_edges = kedro_catalog.load(f'explode_edges_{lin_type}_{lin_type}')
        sim_edges = kedro_catalog.load(f'simplify_edges_{lin_type}_{lin_type}')
        logger.info(f'{lin_type}\t len_exp: {len(exp_edges)}\t len_sim:{len(sim_edges)}\t red:{1-len(sim_edges)/len(exp_edges)}')
        
        
def nx_stats():
    logger.info('summarising nodes, edges, and degree')
    for carrier in ['coal','gas','oil']:
        nodes = kedro_catalog.load(f'community_{carrier}_nodes')
        edges = kedro_catalog.load(f'flow_{carrier}_nx_edges')
        
        G = nx.DiGraph()
        G.add_edges_from([(r[0],r[1],{'z':r[2]}) for r in edges.loc[:,['source','target','z']].values.tolist()])
        hist = nx.degree_histogram(G)
        
        avg_degree = np.dot(hist,list(range(len(hist))))/np.sum(hist)
        logger.info(f'{carrier}:\t n_nodes: {len(nodes)}\t n_edges:{len(edges)}\t avg_deg:{avg_degree}')
    
    

### want total nodes, total edges, average degree

if __name__=="__main__":
    simplify_stats()
    nx_stats()