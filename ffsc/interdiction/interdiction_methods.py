import logging, os, sys, pickle, json, time, yaml
from datetime import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import subprocess

import networkx as nx
import pandas as pd
from math import pi
import numpy as np
from kedro.io import DataCatalog

from ffsc.flow.simplex import network_simplex

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


import multiprocessing as mp

def write_lgf(node_df, edge_df, fpath, run_constrained=False):
    
    if run_constrained:
        C_col='C'
    else:
        C_col='max_C'
    
    with open(fpath,'w') as f:
        f.write('@nodes\n')
        f.write('label supply\n')
        f.writelines([f'{el[0]:d} {el[1]:d}\n' for el in node_df[['index','D']].values.tolist()])
        f.write('\n')

        f.write('@arcs\n')
        f.write('    cost cap\n')
        f.writelines([f'{el[0]:d} {el[1]:d} {el[2]:d} {el[3]:d}\n' for el in edge_df[['source_idx','target_idx','z',C_col]].values.tolist()])
        f.write('\n')
        
        f.write('@attributes\n')
        f.write(f'fpath {os.path.split(fpath)[1]}\n')


def worker_write_lgf(node_key, edge_key, community_params, flow_params, logger):
    """ Each mp worker 
    community_params = {'community_key':str,'community_level':str, 'interdict_keys':str,'run_communities':[]}
    """
    

    if not community_params:
        fname = '_'.join([node_key.split('_')[1],'baseline'])
    else:
        fname = '_'.join([node_key.split('_')[1],str(community_level),'-'.join(interdict_keys)]) # carrier
    savepath = os.path.join(os.getcwd(),'results','interdiction',f'')

    # load kedro catalog and data
    logger.info('Loading data')
    catalog = yaml.load(open(os.path.join(os.getcwd(),'conf','base','catalog.yml'),'r'),Loader=yaml.SafeLoader)
    kedro_catalog = DataCatalog.from_config(catalog)
    
    df_nodes = kedro_catalog.load(node_key)
    df_edges = kedro_catalog.load(edge_key)
    
    
    # write lgf file
    df_nodes = df_nodes.reset_index()
    df_nodes['D'] = -df_nodes['D']
    df_edges['z'] = df_edges['z'].clip(upper=10000).astype(int)
    #df_edges['w'] = np.round(df_edges['z'].max()/(df_edges['z']+1)).astype(int)
    df_edges = pd.merge(df_edges, df_nodes.reset_index()[['index','NODE']], how='left',left_on='source',right_on='NODE').rename(columns={'index':'source_idx'}).drop(columns=['NODE'])
    df_edges = pd.merge(df_edges, df_nodes.reset_index()[['index','NODE']], how='left',left_on='target',right_on='NODE').rename(columns={'index':'target_idx'}).drop(columns=['NODE'])
    df_edges['max_C'] = df_edges['C'].max()
    
    if not community_params: # do baseline only
    
        logger.info('Writing LGF text file')
        write_lgf(df_nodes, df_edges, os.path.join(os.getcwd(),'results','interdiction','lgf',fname+'.lgf'),flow_params['run_constrained'])
        
    else:
        pass
    
    if not community_params:
        return [fname]
    else:
        return True # eventually list fnames
    
    
def call_network_simplex(path):
    
    exec_path = os.path.join(os.getcwd(),'bin','NS.o')
    i_lgf = os.path.join(os.getcwd(),'results','interdiction','lgf',path+'.lgf')
    o_result = os.path.join(os.getcwd(),'results','interdiction','lgf_results',path+'.flow')
    
    process = subprocess.Popen([exec_path, i_lgf, o_result], stdout = subprocess.PIPE)
    process.wait()
    
    return True
    
    
    
def interdiction_baseline_call(params):
    logger=logging.getLogger('MAIN')
    
    # set up logginer
    loggers = {
        'coal':logging.getLogger('COAL-BASELINE'),
        'oil':logging.getLogger('OIL-BASELINE'),
        'gas':logging.getLogger('GAS-BASELINE')
    }
    
    logger.info('setting up params')
    
    # set up worker pool
    logger.info('Setting up pool')
    pool = mp.Pool(params['N_WORKERS'])
    
    # call mp to write the lgf files to use the cpp executable
    worker_params = [
        ('flow_coal_nx_nodes','flow_coal_nx_edges',None,params,loggers['coal']), # coal
        ('flow_oil_nx_nodes','flow_oil_nx_edges',None,params,loggers['oil']), # coal
        ('flow_gas_nx_nodes','flow_gas_nx_edges',None,params,loggers['gas']), # coal
    ]
    
    
    fnames = pool.starmap(worker_write_lgf, worker_params)
    fnames = [item for sublist in fnames for item in sublist]
    print ('fnames', fnames)
    
    # then use the same pool to call the cpp executable
    cpp_results = pool.map(call_network_simplex, fnames)
    
    return []


def load_flow(path, node_df):
    with open(path, 'r') as f:
        lines = f.readlines()
        
    flow = pd.DataFrame([el.strip().split(' ') for el in lines[0:-1]], columns=['source_idx','target_idx','flow']).astype(int)
    flow = pd.merge(flow,node_df[['NODE']],how='left',right_index=True, left_on='source_idx').rename(columns={'NODE':'SOURCE'})
    flow = pd.merge(flow,node_df[['NODE']],how='left',right_index=True, left_on='target_idx').rename(columns={'NODE':'TARGET'})
    
    cost = int(lines[-1].strip().split('=')[1])
    return flow, cost


def interdiction_baseline_parse(df_coal_nodes, df_gas_nodes, df_oil_nodes):
    logger=logging.getLogger('Parse flow results')
    
    paths = {kk:os.path.join(os.getcwd(),'results','interdiction','lgf_results',f'{kk}_baseline.flow') for kk in ['coal','oil','gas']}
    
    cost_records = []
    
    node_dfs = {
        'coal':df_coal_nodes,
        'oil':df_oil_nodes,
        'gas':df_gas_nodes,
    }
    
    flow_dfs = {}
    
    for kk, vv in paths.items():
        if os.path.exists(vv):
            flow_df, cost = load_flow(vv, node_dfs[kk])
            cost_records.append({'carrier':kk,'cost':cost})
            flow_dfs[kk] = flow_df
        else:
            flow_dfs[kk] = pd.DataFrame()
            
    return flow_dfs['coal'], flow_dfs['oil'], flow_dfs['gas'], pd.DataFrame.from_records(cost_records)
        
    
def interdiction_community(df_community_nodes, df_community_edges, df_flow, params):
    if 'COALMINE' in df_community_nodes['NODETYPE'].unique():
        carrier='coal'
    elif 'LNGTERMINAL' in df_community_nodes['NODETYPE'].unique():
        carrier='gas'
    else:
        carrier='oil'
        
    logger = logging.getLogger(f'Interdict network {carrier}')
    
    comm_col = f'comm_{params["community_levels"][carrier]}'
    
    
    df_flow = pd.merge(df_flow, df_community_nodes[['NODE', comm_col]], how='left',left_on='TARGET',right_on='NODE').rename(columns={comm_col:'target_comm'})
    
    # get the top supply communities
    supply_communities = df_flow.loc[df_flow['SOURCE']=='supersource',['target_comm','flow']].groupby('target_comm').sum().sort_values('flow').iloc[-5:].index.values
    
    # get the top demand communities
    demand_communities = df_community_nodes[[comm_col,'D']].groupby(comm_col).sum().sort_values('D').iloc[-5:].index.values
    
    # get the top in between communities
    df_flow = df_flow.set_index(['SOURCE','TARGET'])
    df_community_edges = df_community_edges.set_index(['source','target'])
    df_community_edges = pd.merge(df_community_edges, df_flow[['flow']], how='left',left_index=True, right_index=True)
    tramission_communities = df_community_edges.loc[df_community_edges['source_comm']==df_community_edges['target_comm'],['target_comm','flow']].groupby('target_comm').sum().sort_Values('flow').iloc[-5:].index.values
    
    # mp community interdiction
    pass
    

def interdiction_community_coal(params):
    pass
    
def interdiction_community_oil(params):
    pass

def interdiction_community_gas(params):
    pass
 