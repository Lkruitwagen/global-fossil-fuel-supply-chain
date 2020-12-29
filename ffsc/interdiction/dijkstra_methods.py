import logging, os, sys, pickle, json, time, yaml, glob
from datetime import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import subprocess
from itertools import chain

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

def dijkstra_pypy_pickle(community_nodes, df_edges, params):
    if 'COALMINE' in community_nodes['NODETYPE'].unique():
        carrier = 'coal'
        source_types = ['COALMINE']
    elif 'LNGTERMINAL' in community_nodes['NODETYPE'].unique():
        carrier= 'gas'
        source_types = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        source_types = ['OILFIELD','OILWELL']
        
        
    community_nodes = community_nodes.reset_index().rename(columns={'index':'idx'})
    flow_edges = pd.merge(flow_edges, community_nodes[['idx','NODE']], how='left',left_on='source',right_on='NODE').drop(columns=['NODE']).rename(columns={'idx':'source_idx'})
    flow_edges = pd.merge(flow_edges, community_nodes[['idx','NODE']], how='left',left_on='target',right_on='NODE').drop(columns=['NODE']).rename(columns={'idx':'target_idx'})
    
    G = nx.DiGraph()
    G.add_edges_from([(r[0],r[1], {'z':r[2]}) for r in flow_edges[['source_idx','target_idx','z']].values.tolist()])
    
    supply_nodes = community_nodes.loc[community_nodes['NODETYPE'].isin(source_types),'idx'].values.tolist()
    target_nodes = community_nodes.loc[community_nodes['NODETYPE'].isin(source_types),'idx'].values.tolist()
    
    pickle.dump(supply_nodes, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'supply_nodes.pkl'),'wb'),protocol=2)
    pickle.dump(target_nodes, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'target_nodes.pkl'),'wb'),protocol=2)
    pickle.dump(G._succ,open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'edges.pkl'),'wb'),protocol=2)
    
    return []
        
def dijkstra_pypy_paths(community_nodes, df_edges, params):
    if 'COALMINE' in community_nodes['NODETYPE'].unique():
        carrier = 'coal'
        source_types = ['COALMINE']
    elif 'LNGTERMINAL' in community_nodes['NODETYPE'].unique():
        carrier= 'gas'
        source_types = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        source_types = ['OILFIELD','OILWELL']
        
    


def dijkstra_prep_paths(community_nodes, df_edges, params):
    
    
    if 'COALMINE' in community_nodes['NODETYPE'].unique():
        carrier = 'coal'
        source_types = ['COALMINE']
    elif 'LNGTERMINAL' in community_nodes['NODETYPE'].unique():
        carrier= 'gas'
        source_types = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        source_types = ['OILFIELD','OILWELL']

    # instantiate the graph
    G = nx.DiGraph()
    G.add_edges_from([(r[0],r[1], {'z':r[2]}) for r in df_edges[['source','target','z']].values.tolist()])
    
    # get the dijkstra trees for each one
    for ii_s,sn in enumerate(community_nodes.loc[community_nodes['NODETYPE'].isin(source_types),:].groupby(f'comm_{params["COMMUNITY_LEVEL"][carrier]}').nth(0)['NODE'].values.tolist()):

        print (ii_s, sn)
        if not os.path.exists(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',sn+'.json')):

            cost, path = nx.single_source_dijkstra(G, sn, cutoff=None, weight='z')

            path = {kk:vv for kk,vv in cost.items() if 'CITY' in kk or 'POWERSTATION' in kk}

            json.dump(path, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier, sn+'.json'),'w'))
        else:
            print ('exists', os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,sn+'.json'))
        
    return []


def dijkstra_parse_paths_mpworker(node_communities, json_paths, params,logger):
    
    node_subdict = {kk:[] for kk in node_communities.keys()}
    
    for ii_j, json_path in enumerate(json_paths):
        
        logger.info(f'parsing {ii_j}, {json_path}')
        
        title = os.path.splitext(os.path.split(json_path)[1])[0]
    
        paths = json.load(open(json_path,'r'))
        
        paths = {kk:{'paths':vv, 'source_comm':node_communities[title],'target_comm':node_communities[kk]} for kk, vv in paths.items()}
        
        for kk, vv in paths.items():
            for NODE in vv['paths']:
                if node_communities[NODE]!=vv['source_comm']:
                    node_subdict[NODE].append((vv['source_comm'],vv['target_comm']))
        
        for kk in node_subdict.keys():
            node_subdict[kk] = list(set(node_subdict[kk]))
    
    return node_subdict

def dijkstra_parse_paths(community_nodes, params):
    
    if 'COALMINE' in community_nodes['NODETYPE'].unique():
        carrier = 'coal'
    elif 'LNGTERMINAL' in community_nodes['NODETYPE'].unique():
        carrier= 'gas'
    else:
        carrier='oil'
        
    logger = logging.getLogger(f'Parse paths {carrier}')
    
    comm_col = f'comm_{params["COMMUNITY_LEVEL"][carrier]}'

    json_paths = glob.glob(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'*'))
    
    logger.info(f'len json_paths: {len(json_paths)}')
    
    node_dict = {kk:[] for kk in community_nodes['NODE'].values.tolist()}
    
    node_communities = {r[0]:r[1] for r in community_nodes[['NODE',comm_col]].values.tolist()}
    
    pool = mp.Pool(N_WORKERS)
    
    CHUNK = (len(json_paths))//N_WORKERS+1
    
    print ('chunk',CHUNK,'nworkers',N_WORKERS)
    
    mp_params = [[node_communities, json_paths[ii_w*CHUNK:(ii_w+1)*CHUNK], params, logging.getLogger(f'mp parsepaths {ii_w}')] for ii_w in range(N_WORKERS)]
    
    node_communities_populated = pool.starmap(dijkstra_parse_paths_mpworker, mp_params)
    
    for kk in node_dict.keys():
        node_dict[kk] = list(set(chain.from_iterable(pp[kk] for pp in node_communities_populated)))
        
    return node_dict


def reachable_worker(json_path_comms, all_comms, ii_w):
    
    reachable_subdict = {comm:[] for comm in list(set(all_comms))}
    
    for ii_j, (json_path, comm) in enumerate(json_path_comms):
        
        print(f'{ii_w} parsing {ii_j}, {json_path}')
        
        # load the json
        paths = json.load(open(json_path,'r'))
        
        reachable_subdict[comm] = list(paths.keys())
        
    return reachable_subdict


def dijkstra_filter_reachable(community_nodes, params):
    """
    
    returns: pickle(source_comm:[reachable_targets])
    """
    
    if 'COALMINE' in community_nodes['NODETYPE'].unique():
        carrier = 'coal'
    elif 'LNGTERMINAL' in community_nodes['NODETYPE'].unique():
        carrier= 'gas'
    else:
        carrier='oil'
        
    logger = logging.getLogger(f'Parse paths {carrier}')
    
    comm_col = f'comm_{params["COMMUNITY_LEVEL"][carrier]}'
        
    json_paths = glob.glob(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'*'))
    titles = [os.path.splitext(os.path.split(json_path)[1])[0] for json_path in json_paths]
    
    logger.info(f'len json_paths: {len(json_paths)}')
    
    community_nodes = community_nodes.set_index('NODE')
    
    all_comms = community_nodes.loc[titles,comm_col].values.tolist()

    CHUNK = len(json_paths)//N_WORKERS + 1
    
    mp_params = [
        [list(zip(json_paths,all_comms))[ii_w*CHUNK:(ii_w+1)*CHUNK], all_comms, ii_w]
        for ii_w in range(N_WORKERS)
    ]
    
    pool = mp.Pool(N_WORKERS)
    
    reachable_dicts = pool.starmap(reachable_worker, mp_params)
    
    reachable_dict = {comm:[] for comm in list(set(all_comms))}
    
    for dd in reachable_dicts:
        for comm, ll in dd.items():
            reachable_dict[comm]+=ll
        
    reachable_dict = pd.DataFrame({kk:[vv] for kk, vv in reachable_dict.items()}).T.rename(columns={0:'reachable_targets'})
    reachable_df.index = reachable_df.index.astype(int)
    
    return reachable_dict


def dijkstra_shortest_allpairs(community_nodes, df_edges, nodeflags, reachable_dict, params):
    
    """
    
    returns: pickle({source:{target:{dist:shortest_z, path:path} for target in reachable_targets} for source in sources})
    ... maybe parallelise this also.
    """    
    
    if 'COALMINE' in community_nodes['NODETYPE'].unique():
        carrier = 'coal'
        source_types = ['COALMINE']
    elif 'LNGTERMINAL' in community_nodes['NODETYPE'].unique():
        carrier= 'gas'
        source_types = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        source_types = ['OILFIELD','OILWELL']
        
    logger = logging.getLogger(f'Dijkstra shortest {carrier}')
    
    comm_col = f'comm_{params["COMMUNITY_LEVEL"][carrier]}'
    
    logger.info('prepping')
    # cast to dict from pandas
    reachable_dict = reachable_dict['reachable_targets'].to_dict()
    nodeflags['nodeflags'] = nodeflags['nodeflags'].apply(lambda ll: [tuple(el) for el in ll])
    nodeflags = nodeflags['nodeflags'].to_dict()
      
    logger.info('Making Graph')
    # instantiate the graph
    G = nx.DiGraph()
    G.add_edges_from([(r[0],r[1], {'z':r[2]}) for r in df_edges[['source','target','z']].values.tolist()])
    
    # get source nodes
    source_nodes = community_nodes.loc[community_nodes['NODETYPE'].isin(source_types),'NODE'].values.tolist()
    
    community_nodes = community_nodes.set_index('NODE')[comm_col].to_dict()
    
    dijkstra_pairs = {}
    
    # get the dijkstra trees for each one
    for ii_s,sn in enumerate(source_nodes):
        
        tic = time.time()
        
        dijkstra_pairs[sn] = {}
        
        SOURCE_COMM = community_nodes[sn]
        BIG_ERR = 10000000
        
        reachable_targets = reachable_dict[SOURCE_COMM]
        logger.info(f'doing supply node {ii_s}: {sn} \t {len(reachable_targets)} reachable targets')
        
        # heuristic needs source_comm, declare in loop
        def heuristic(neighbour,target):
            
            """ meant to estimate the distance between the source and the target"""
            
            if community_nodes[neighbour]==community_nodes[target]:
                # if neighbour in same comm, don't leave
                #print ('same')
                return 0
            elif (SOURCE_COMM,community_nodes[target]) in nodeflags[neighbour]:
                # if neighbour is on the quickest path between the two communities
                #print ('quickpath')
                return 0
            else:
                #print ('meow')
                return BIG_ERR
        
        for target in reachable_targets:
            toc = time.time()
            
            path = nx.astar_path(G, sn, target, heuristic=heuristic, weight="z")
            dist = sum([G[u][v]['z'] for u, v in zip(path[:-1], path[1:])])
            
            dijkstra_pairs[sn][target] = {'dist':dist, 'path':path}
            print (sn, target,time.time()-toc)
        
    return dijkstra_pairs


def differential_fill(community_nodes, dijkstra_pairs, reachable_targets, supply_alpha):
    """
    supply_alpha: {source:float}
    differential fill algo, marginal supply cost of the form: alpha*flow + dist
    (supply cost: alpha/2*flow**2 + dist*flow)
    reachable_targets: pickle(source_comm:[reachable_targets])
    reverse_reachable: {target:[list_sources]}
    dijkstra pairs: {source:{target:{dist:shortest_z, path:path} for target in reachable_targets} for source in sources}
    """    
    
    # set up dfs - source is column, target is index
    
    df_z_ini = pd.DataFrame({source:{target:vv2['dist'] for target,vv2 in vv1.items()} for source,vv1 in disjkstra_pairs})
    df_z = df_z_ini.copy()
    df_flow = df_z.copy()
    df_flow[~df_flow.isna()]=0

    
    if 'COALMINE' in community_nodes['NODETYPE'].unique():
        carrier = 'coal'
        source_types = ['COALMINE']
    elif 'LNGTERMINAL' in community_nodes['NODETYPE'].unique():
        carrier= 'gas'
        source_types = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        source_types = ['OILFIELD','OILWELL']
        
    logger = logging.getLogger(f'Dijkstra shortest {carrier}')
    
    comm_col = f'comm_{params["COMMUNITY_LEVEL"][carrier]}'
    
    df_dem = pd.Series({r[0]:r[1] for r in community_nodes[['NODE','D']].values.tolist()})  
    
    reverse_reachable = pd.DataFrame([[el] for el in reachable_targets.values()], columns=['idx'], index=reachable_targets.keys()) \
                            .reset_index() \
                            .explode('idx') \
                            .groupby('idx')['index'] \
                            .apply(list).to_dict()
    
    ### loop:
    ITER = 15
    STEP = 10
    for ii in range(ITER): # while any demand is left while df_dem.sum()>0
        tic = time.time()
        print(f'{ii} \t new iter')
        
        
        # add_flow should be a vector in dimension of the target
        add_flow = pd.Series(STEP, index=df_dem.index).combine(df_dem, min, 0)
        
        # add the flow to the min cost pair
        df_flow.unstack().loc[[(v,k) for k,v in df_z.T.idxmin().iteritems()]] += add_flow.values
        
        # subtrace the flow from outstanding demand
        df_dem -= add_flow
        
        # update df_cost
        df_z = df_z_ini + df_alpha * df_flow.sum()
        
    return df_flow, df_z


#def differential_fill_iter(community_nodes, dijkstra_pairs, reachable_targets, supply_alpha):