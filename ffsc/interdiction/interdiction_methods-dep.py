import logging, os, sys, pickle, json, time, yaml
from datetime import datetime as dt
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
import pandas as pd
from math import pi
import numpy as np
from kedro.io import DataCatalog

from ffsc.flow.simplex import network_simplex

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


import multiprocessing as mp

def write_lgf(node_df, edge_df, fpath):
    
    with open(fpath,'w') as f:
        f.write('@nodes\n')
        f.write('label supply')
        f.writelines([f'{el[0]:d} {el[1]:d}\n' for el in node_df[['index','D']].values.tolist()])
        f.write('\n')

        f.write('@edges\n')
        f.write('    cost cap\n')
        f.writelines([f'{el[0]:d} {el[1]:d} {el[2]:d} {el[3]:d}\n' for el in edge_df[['source_idx','target_idx','w','C']].values.tolist()])
        f.write('\n')
        
        f.write('@attributes\n')
        f.write(f'fpath {os.path.split(fpath)[1]}\n')


def interdiction_worker(node_key, edge_key, community_key, community_level, interdict_keys, logger):

    if not interdict_keys:
        interdict_keys = ['NONE']
        
    fname = '_'.join([node_key.split('_')[1],str(community_level),'-'.join(interdict_keys)]) # carrier
    savepath = os.path.join(os.getcwd(),'results','interdiction',f'')

    # load kedro catalog and data
    logger.info('Loading data')
    catalog = yaml.load(open(os.path.join(os.getcwd(),'conf','base','catalog.yml'),'r'),Loader=yaml.SafeLoader)
    kedro_catalog = DataCatalog.from_config(catalog)
    
    df_nodes = kedro_catalog.load(node_key)
    df_edges = kedro_catalog.load(edge_key)
    
    
    # write lgf file
    node_df = node_df.reset_index()
    edge_df['w'] = np.round(edge_df['z'].max()/(edge_df['z']+1)).astype(int)
    edge_df = pd.merge(edge_df, node_df.reset_index()[['index','NODE']], how='left',left_on='source',right_on='NODE').rename(columns={'index':'source_idx'}).drop(columns=['NODE'])
    edge_df = pd.merge(edge_df, node_df.reset_index()[['index','NODE']], how='left',left_on='target',right_on='NODE').rename(columns={'index':'target_idx'}).drop(columns=['NODE'])
    
    logger.info('Writing LGF text file')
    write_lgf(node_df, edge_df, os.path.join(os.getcwd(),'results','interdiction','lgf',fname+'.lgf'))
    
    
    
    """ NetworkX (dep)
    # construct graph
    logger.info('Making Graph')
    G = nx.from_pandas_edgelist(df_edges, edge_attr = ['C','z'], create_using=nx.DiGraph)
    nx.set_node_attributes(G, df_nodes.set_index('NODE').to_dict('index'))
    
    
    # do interdiction
    # ... nothing for now.
    
    
    
    logger.info('Running simplex')
    if flow_parameters['run_constrained']:
        flow_cost, flow_dict = network_simplex(G, demand='D', capacity='C', weight='z', logger=logger)
    else:
        flow_cost, flow_dict = network_simplex(G, demand='D', capacity='na', weight='z', logger=logger)
    
    with open(os.path.join(os.getcwd(),'results','flow','flow_cost_log.txt'),'a') as f:
        f.write(','.join([dt.now().isoformat(),fname,str(flow_cost)]))
    """

    #print ('flow cost',flow_cost)
    #print (flow_dict)

    pickle.dump(flow_cost, open('./flow_cost.pkl','wb'))
    pickle.dump(flow_dict, open('./flow_dict.pkl','wb'))
    
    return True
    
    
    
    
def interdiction_baseline(params):
    logger=logging.getLogger('MAIN')
    
    # set up logginer
    loggers = {
        'coal':logging.getLogger('COAL-BASELINE'),
        'oil':logging.getLogger('OIL-BASELINE'),
        'gas':logging.getLogger('GAS-BASELINE')
    }
    
    logger.info('setting up params')
    # mp baseline x3
    worker_params = [
        ('flow_coal_nx_nodes','flow_coal_nx_edges',None,None,None,loggers['coal']), # coal
        ('flow_oil_nx_nodes','flow_oil_nx_edges',None,None,None,loggers['oil']), # coal
        ('flow_gas_nx_nodes','flow_gas_nx_edges',None,None,None,loggers['gas']), # coal
    ]
    
    logger.info('Setting up pool')
    pool = mp.Pool(params['N_WORKERS'])
    
    pool.starmap(interdiction_worker, worker_params)
    
    return []
    
    
    
def interdiction_community(node_key, edge_key, community_key, params):
    
    # mp community interdiction
    pass
    

def interdiction_community_coal(params):
    pass
    
def interdiction_community_oil(params):
    pass

def interdiction_community_gas(params):
    pass
 