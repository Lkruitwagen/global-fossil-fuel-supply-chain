import logging, os, sys, pickle, json, time, yaml, glob
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

N_WORKERS=6

# total cost overflowing max int: 2147483647

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
    community_params = {'comm_col':comm_col, 'carrier':carrier, interdict_communities':[{'type':<'supply','demand','transmission'>,'comm_id':comm} for comm in top_supply_communities]}
    """
    

    if not community_params:
        fname = '_'.join([node_key.split('_')[1],'baseline'])
    else:
        fnames = []

    # load kedro catalog and data
    logger.info('Loading data')
    catalog = yaml.load(open(os.path.join(os.getcwd(),'conf','base','catalog.yml'),'r'),Loader=yaml.SafeLoader)
    kedro_catalog = DataCatalog.from_config(catalog)
    
    df_nodes = kedro_catalog.load(node_key)
    df_edges = kedro_catalog.load(edge_key)
    
    
    
    
    supply_type = {'coal':['COALMINE'],'oil':['OILFIELD','OILWELL'],'gas':['OILFIELD','OILWELL']}
    
    if not community_params: # do baseline only
        
        # write lgf file
        df_nodes = df_nodes.reset_index()
        df_nodes['D'] = -df_nodes['D']
        df_edges['z'] = (df_edges['z']/10).astype(int)
        #df_edges['w'] = np.round(df_edges['z'].max()/(df_edges['z']+1)).astype(int)
        df_edges = pd.merge(df_edges, df_nodes.reset_index()[['index','NODE']], how='left',left_on='source',right_on='NODE').rename(columns={'index':'source_idx'}).drop(columns=['NODE'])
        df_edges = pd.merge(df_edges, df_nodes.reset_index()[['index','NODE']], how='left',left_on='target',right_on='NODE').rename(columns={'index':'target_idx'}).drop(columns=['NODE'])
        df_edges['max_C'] = df_edges['C'].max()*2
    
        logger.info('Writing LGF text file')
        write_lgf(df_nodes, df_edges, os.path.join(os.getcwd(),'results','interdiction','lgf',fname+'.lgf'),flow_params['run_constrained'])
        
    else: # do the interdictions
        
        for comm_dd in community_params['interdict_communities']:
            fname = '_'.join([node_key.split('_')[1],community_params['comm_col'],comm_dd['type'],'id',str(comm_dd['comm_id'])])
            logger.info(f'prepping {fname}')
            
            comm_nodes = df_nodes.copy()
            comm_edges = df_edges.copy()
            
            comm_nodes['D'] = -1*comm_nodes['D']
            comm_edges['z'] = (comm_edges['z']/10).astype(int)
            
            if comm_dd['type']=='supply':
                comm_nodes = comm_nodes[~((comm_nodes['NODETYPE'].isin(supply_type[community_params['carrier']])) & (comm_nodes[community_params['comm_col']]==comm_dd['comm_id']))]

            elif comm_dd['type']=='demand':
                # don't want to remove these nodes, jsut set their demand to 0
                #comm_nodes = comm_nodes[~((comm_nodes['NODETYPE'].isin(['POWERSTATION','CITY'])) & (comm_nodes[community_params['comm_col']]==comm_dd['comm_id']))]
                
                comm_nodes.loc[((comm_nodes['NODETYPE'].isin(['POWERSTATION','CITY'])) & (comm_nodes[community_params['comm_col']]==comm_dd['comm_id'])),'D'] = 0
                

            elif comm_dd['type']=='transmission':
                #print('trans before', comm_dd['comm_id'])
                #print (comm_nodes)
                comm_nodes = comm_nodes[~(comm_nodes[community_params['comm_col']]==comm_dd['comm_id'])]
                #print('trans after', comm_dd['comm_id'])
                #print (comm_nodes)
                
            
            # rm from edges
            comm_edges = comm_edges[comm_edges['source'].isin(comm_nodes['NODE']) & comm_edges['target'].isin(comm_nodes['NODE'])]
            
            # add supersource
            supplies = comm_nodes.loc[comm_nodes['NODETYPE'].isin(supply_type[community_params['carrier']]),'NODE'].values.tolist()
            
            comm_nodes = comm_nodes.append(
                pd.DataFrame(
                    {
                        'D':[-comm_nodes['D'].sum()],
                        'NODE':['supersource'],
                    }
                ), ignore_index=True
            ).reset_index()
            
            comm_edges = comm_edges.append(
                pd.DataFrame(
                    {'source':['supersource']*len(supplies),
                     'target':supplies,
                     'z':[0]*len(supplies),
                     'C':[comm_edges['C'].max()]*len(supplies)
                    }
                ), ignore_index=True
            )
            
            if comm_dd['type'] in ['transmission','demand']:
                # without e.g. certain cities, perhaps downstream cities or pss are unreachable.
                
                
                logger.info('Checking transmission connectivity')
                ## make sure edges are reachable
                G = nx.DiGraph()
                G.add_edges_from([(r[0],r[1]) for r in comm_edges[['source','target']].values.tolist()])
                
                tt = nx.single_source_dijkstra(G, 'supersource')              
                
                comm_edges = comm_edges[comm_edges['target'].isin(tt[0].keys()) & comm_edges['source'].isin(tt[0].keys())]
                comm_nodes = comm_nodes[comm_nodes['NODE'].isin(tt[0].keys())]
                
                
                comm_nodes.loc[comm_nodes['NODE']=='supersource','D'] = -1* comm_nodes.loc[comm_nodes['NODE']!='supersource','D'].sum()
                
            
            # merge indexes
            comm_edges = pd.merge(comm_edges, comm_nodes[['index','NODE']], how='left',left_on='source', right_on='NODE').rename(columns={'index':'source_idx'})
            comm_edges = pd.merge(comm_edges, comm_nodes[['index','NODE']], how='left',left_on='target', right_on='NODE').rename(columns={'index':'target_idx'})
            comm_edges['max_C'] = comm_edges['C'].max()*2
            
            logger.info(f'writing {fname}')
            write_lgf(
                comm_nodes[['index','D']].astype(int), 
                comm_edges[['source_idx','target_idx','z','C','max_C']].astype(int), 
                os.path.join(os.getcwd(),'results','interdiction','lgf',fname+'.lgf'),
                flow_params['run_constrained']
            )
        
            fnames.append(fname)

    
    if not community_params:
        return [fname]
    else:
        return fnames # eventually list fnames
    
    
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
            if cost<0:
                cost = cost+2**32
            cost_records.append({'carrier':kk,'cost':cost})
            flow_dfs[kk] = flow_df
        else:
            flow_dfs[kk] = pd.DataFrame()
            
    return flow_dfs['coal'], flow_dfs['oil'], flow_dfs['gas'], pd.DataFrame.from_records(cost_records)


def interdiction_community_parse(df_community_nodes):
    # get carrier:
    if 'COALMINE' in df_community_nodes['NODETYPE'].unique():
        carrier='coal'
        supply_type=['COALMINE']
    elif 'LNGTERMINAL' in df_community_nodes['NODETYPE'].unique():
        carrier='gas'
        supply_type=['OILFIELD','OILWELL']
    else:
        carrier='oil'
        supply_type=['OILFIELD','OILWELL']
        
    logger=logging.getLogger(f'parse_interdictions_{carrier}')
    
    paths = glob.glob(os.path.join(os.getcwd(),'results','interdiction','lgf_results',f'{carrier}_comm_*.flow'))
    
    #supplies = df_community_nodes.loc[df_community_nodes['NODETYPE'].isin(supply_type),'NODE'].values.tolist()
            
    df_community_nodes = df_community_nodes.append(
        pd.DataFrame(
            {
                'D':[-df_community_nodes['D'].sum()],
                'NODE':['supersource'],
            }
        ), ignore_index=True
    )
    
    cost_records = []
    
    for pp in paths:
        logger.info(f'loading {pp}')
        _id = '_'.join(os.path.split(os.path.splitext(pp)[0])[1].split('_')[3:])
        flow_df, cost = load_flow(pp, df_community_nodes)
        if cost<0:
            cost = cost+2**32
        cost_records.append({'id':_id,'path':pp,'cost':cost})
        
        flow_df.to_parquet(os.path.splitext(pp)[0]+'.pqt')
        
    cost_df = pd.DataFrame.from_records(cost_records)
    print ('cost_df')
    print (cost_df)
    
    cost_df.to_csv(os.path.join(os.getcwd(),'results','interdiction',f'{carrier}_interdiction.csv'))
            
            
    return []
            
    
    
    
    
        
    
def interdiction_community(df_community_nodes, df_community_edges, df_community, df_flow, params):
    if 'COALMINE' in df_community_nodes['NODETYPE'].unique():
        carrier='coal'
    elif 'LNGTERMINAL' in df_community_nodes['NODETYPE'].unique():
        carrier='gas'
    else:
        carrier='oil'
        
    node_keys = {'coal':'community_coal_nodes','oil':'community_oil_nodes','gas':'community_gas_nodes'}
    edge_keys = {'coal':'community_coal_edges','oil':'community_oil_edges','gas':'community_gas_edges'}
        
    logger = logging.getLogger(f'Interdict network {carrier}')
    
    comm_col = f'comm_{params["COMMUNITY_LEVEL"][carrier]}'
    
    
    df_flow = pd.merge(df_flow, df_community_nodes[['NODE', comm_col]], how='left',left_on='TARGET',right_on='NODE').rename(columns={comm_col:'target_comm'})
    
    # get the top supply communities
    top_supply_communities = df_flow.loc[df_flow['SOURCE']=='supersource',['target_comm','flow']].groupby('target_comm').sum().sort_values('flow').iloc[-5:].index.values
    
    # get the top demand communities
    top_demand_communities = df_community_nodes[[comm_col,'D']].groupby(comm_col).sum().sort_values('D').iloc[-5:].index.values
    
    # get the top in between communities
    transmission_communities = df_community[(df_community['supply']==False) & (df_community['demand']==False)].index.values
    df_flow = df_flow.rename(columns={'SOURCE':'source','TARGET':'target'}).set_index(['source','target'])
    df_community_edges = df_community_edges.set_index(['source','target'])
    print ('df_flow')
    print (df_flow)
    print ('community_edges')
    print (df_community_edges)
    df_community_edges = pd.merge(df_community_edges, df_flow[['flow']], how='left',left_index=True, right_index=True)
    top_transmission_communities = df_community_edges.loc[df_community_edges['target_comm'].isin(transmission_communities) & (df_community_edges['source_comm']==df_community_edges['target_comm']),['target_comm','flow']].groupby('target_comm').sum().sort_values('flow').iloc[-5:].index.values
    
    print ('top communities')
    print (top_supply_communities, top_demand_communities, top_transmission_communities)
    
    logger.info('Running MP lgf writer')
    # one each
    community_params = [
        {'comm_col':comm_col,'carrier':carrier, 'interdict_communities':[{'type':'supply','comm_id':comm} for comm in top_supply_communities]+[{'type':'transmission','comm_id':comm}  for comm in top_transmission_communities[2:4]]},
        {'comm_col':comm_col,'carrier':carrier, 'interdict_communities':[{'type':'demand','comm_id':comm}  for comm in top_demand_communities]+[{'type':'transmission','comm_id':comm}  for comm in top_transmission_communities[4:]]},
        {'comm_col':comm_col,'carrier':carrier, 'interdict_communities':[{'type':'transmission','comm_id':comm}  for comm in top_transmission_communities[0:2]]}
    ]
    
    
    worker_params = [
        [node_keys[carrier], edge_keys[carrier], community_params[0], params, logging.getLogger(f'{carrier}_writer-1')],
        [node_keys[carrier], edge_keys[carrier], community_params[1], params, logging.getLogger(f'{carrier}_writer-2')],
        [node_keys[carrier], edge_keys[carrier], community_params[2], params, logging.getLogger(f'{carrier}_writer-3')],
    ]
    
    pool = mp.Pool(N_WORKERS)
    
    fnames = pool.starmap(worker_write_lgf, worker_params)
    fnames = [item for sublist in fnames for item in sublist]
    logger.info('Finished writing lgf files')
    print ('fnames', fnames)
    
    # then use the same pool to call the cpp executable
    logger.info('Call simplex')
    cpp_results = pool.map(call_network_simplex, fnames)
    
    # mp community interdiction
    return []
    

def interdiction_community_coal(params):
    pass
    
def interdiction_community_oil(params):
    pass

def interdiction_community_gas(params):
    pass
 