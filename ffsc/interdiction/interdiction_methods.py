import logging, os, sys, pickle, json, time, yaml, glob
from datetime import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import subprocess
from itertools import chain
from tqdm import tqdm

import networkx as nx
import pandas as pd
from math import pi
import numpy as np
from kedro.io import DataCatalog

from ffsc.flow.simplex import network_simplex
from ffsc.interdiction.gp import *
from ffsc.interdiction.dijkstra_methods import differential_fill

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

N_WORKERS=8

# total cost overflowing max int: 2147483647

import multiprocessing as mp


def sds_demand_counterfactual(iso2, df_sds, df_nodes, node_iso2, edge_df, params, dijkstra_min_adj):
    """
    Make flow df for SDS case
    """
    
    ## turn the dijkstra results into flow dfs
    if 'COALMINE' in df_nodes['NODETYPE'].unique():
        carrier = 'coal'
        source_types = ['COALMINE']
    elif 'LNGTERMINAL' in df_nodes['NODETYPE'].unique():
        carrier= 'gas'
        source_types = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        source_types = ['OILFIELD','OILWELL']
    
    logger = logging.getLogger(f'SDS_counterfact_{carrier}')
    
    node_iso2 = node_iso2.groupby('NODE').nth(0).reset_index()
    
    # merge iso2 -> correct for bad geometries
    df_nodes = pd.merge(df_nodes, node_iso2, how='left', left_on='NODE',right_on='NODE')
    df_nodes.loc[df_nodes['NODE'].isin(['CITY_218','CITY_219','CITY_220']),'iso2']='AR'
    df_nodes.loc[df_nodes['NODE'].isin(['CITY_3565','POWERSTATION_8344']), 'iso2']='DK'
    df_nodes.loc[df_nodes['NODE'].isin(['CITY_4635','CITY_4636','POWERSTATION_10692']), 'iso2'] = 'GA'
    df_nodes.loc[df_nodes['NODE'].isin(['POWERSTATION_27208','POWERSTATION_26808']), 'iso2'] = 'US'
    df_nodes.loc[df_nodes['NODE'].isin(['POWERSTATION_13119']), 'iso2'] = 'IE'
    df_nodes.loc[df_nodes['NODE'].isin(['POWERSTATION_5117']), 'iso2'] = 'CN'
    df_nodes.loc[df_nodes['NODE'].isin(['POWERSTATION_7847']),'iso2'] = 'CY'
    df_nodes.loc[df_nodes['NODE'].isin(['POWERSTATION_14316', 'POWERSTATION_14317', 'POWERSTATION_14321']),'iso2'] = "'NA"
    
    # merge region
    df_nodes = pd.merge(df_nodes, iso2[['iso2','region_weo2019']], how='left', left_on='iso2',right_on='iso2')
    
    
    # get reduction in final and power energy
    df_sds['REDUCTION-FIN'] = (df_sds[f'2040SDS-TPED-{carrier.upper()}']-df_sds[f'2040SDS-POWER-{carrier.upper()}'])/(df_sds[f'2018-TPED-{carrier.upper()}'] -  df_sds[f'2018-POWER-{carrier.upper()}'])
    df_sds['REDUCTION-POWER'] = df_sds[f'2040SDS-POWER-{carrier.upper()}'] / df_sds[f'2018-POWER-{carrier.upper()}']
    
    ## merge reduction onto nodes
    df_nodes = pd.merge(df_nodes,df_sds[['REGION','REDUCTION-FIN','REDUCTION-POWER']], how='left',left_on='region_weo2019',right_on='REGION')
    
    # take reduction out of demand
    df_nodes.loc[df_nodes['NODETYPE']=='CITY','D'] = df_nodes.loc[df_nodes['NODETYPE']=='CITY','D']* df_nodes.loc[df_nodes['NODETYPE']=='CITY','REDUCTION-FIN']
    df_nodes.loc[df_nodes['NODETYPE']=='POWERSTATION','D'] = df_nodes.loc[df_nodes['NODETYPE']=='POWERSTATION','D'] * df_nodes.loc[df_nodes['NODETYPE']=='POWERSTATION','REDUCTION-POWER']
    
    # accomodate antarcitca
    df_nodes.loc[df_nodes['iso2']=='AQ','D'] = 0
    
    print (df_nodes)
    print (df_nodes[df_nodes['D'].isin([np.nan, np.inf, -np.inf])])
    
    # round and int demand
    df_nodes['D'] = np.round(df_nodes['D'],0).astype(int)
    
    logger.info('loading run data')
    run_data = pickle.load(open(params['flowfill_run'][carrier],'rb'))  
    SCALE_FACTOR = run_data['SCALE_FACTOR']
    STEP_INI = run_data['STEP_INI']
    GAMMA = run_data['GAMMA']
    df_alpha = run_data['ALPHA']

    
    logger.info('running fill algo')
    ii_w, df_flow, df_z = differential_fill(df_nodes, dijkstra_min_adj, df_alpha, STEP_INI, GAMMA,SCALE_FACTOR, params, logging.getLogger('ini_diff_fill'), None)
    print ('df flow')
    print (df_flow)
    
    logger.info('prepping dfs')
    sources = df_flow.sum()[df_flow.sum()>0].index.tolist()
    
    df_nodes = df_nodes.reset_index().rename(columns={'index':'idx'}).set_index('NODE')
    
    edge_df['flow']=0
    edge_df = pd.merge(edge_df, df_nodes[['idx']], how='left',left_on='source',right_index=True).rename(columns={'idx':'source_idx'})
    edge_df = pd.merge(edge_df, df_nodes[['idx']], how='left',left_on='target',right_index=True).rename(columns={'idx':'target_idx'})
    edge_df = edge_df.set_index(['source_idx','target_idx'])
    
    logger.info('filling flow paths')
    for source in tqdm(sources, desc='adding flow'):
        source_idx = df_nodes.at[source,'idx']
        if carrier=='oil':
            paths = pickle.load(open(f'/paths/{carrier}_join/{source_idx}.pkl','rb'))
        else:
            paths = pickle.load(open(f'/paths/{carrier}/{source_idx}.pkl','rb'))
        for dest in df_flow.loc[df_flow[source]>0,source].index.values.tolist():
            dest_idx = df_nodes.at[dest,'idx']
            add_flow = df_flow.at[dest,source]
            #print('add flow',add_flow)
            node_path_list = paths[dest_idx]
            node_path_tups = list(zip(node_path_list[0:-1],node_path_list[1:]))
            if carrier=='oil':
                # make sure the idxs arent the same
                node_path_tups = [nn for nn in node_path_tups if nn[0]!=nn[1]]
            #print ('node_path_tups',node_path_tups)
            edge_df.loc[node_path_tups,'flow']+=add_flow

    return edge_df.reset_index().drop(columns=['source_idx','target_idx'])
    
    
    
    
def interdict_supply(node_df, edge_df, params, dijkstra_min_adj):
    """
    For all supply nodes, try doubling the quadratic term
    """
    
    ## turn the dijkstra results into flow dfs
    if 'COALMINE' in node_df['NODETYPE'].unique():
        carrier = 'coal'
        source_types = ['COALMINE']
    elif 'LNGTERMINAL' in node_df['NODETYPE'].unique():
        carrier= 'gas'
        source_types = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        source_types = ['OILFIELD','OILWELL']
    
    logger = logging.getLogger(f'Dijkstra_post_{carrier}')
    
    logger.info('loading run data')
    run_data = pickle.load(open(params['flowfill_run'][carrier],'rb'))  
    SCALE_FACTOR = run_data['SCALE_FACTOR']
    STEP_INI = run_data['STEP_INI']
    GAMMA = run_data['GAMMA']
    df_alpha = run_data['ALPHA']
    
    logger.info('running fill algo')
    ii_w, df_flow, df_z = differential_fill(node_df, dijkstra_min_adj, df_alpha, STEP_INI, GAMMA,SCALE_FACTOR, params, logging.getLogger('ini_diff_fill'), None)
    
    pool = mp.Pool(N_WORKERS)
    
    for ITER in range(df_alpha.shape(0)//N_WORKERS+1):
        differential_fill_params = []
        
        # prep params, multiplying a supply curve every time
        for ii_w in range(N_WORKERS):
            df = df_alpha.copy()
            df.iloc[ITER+ii_w,:] = df.iloc[ITER+ii_w,:]*2
            differential_fill_params.append(
                [
                    community_nodes,
                    dijkstra_mincost_adj,  
                    df,
                    STEP_INI,
                    GAMMA,
                    SF, # try some random scaling factors between 10**2 and 10**6
                    params,
                    logging.getLogger(f'dijkstra_fill_worker_{ii_w}'),
                    ii_w,
                ]
            )
            
        results = pool.starmap(differential_fill, differential_fill_params)

        fitness = {}
        for ii_w, df_flow, df_z in results:
            pickle.dump(df_flow, open(os.path.join(os.getcwd(),'results','interdiction','supply',f'flow_{ITER*N_WORKERS+ii_w}.pkl'),'wb'))
            pickle.dump(df_z, open(os.path.join(os.getcwd(),'results','interdiction','supply',f'z_{ITER*N_WORKERS+ii_w}.pkl'),'wb'))


    return []
    




def call_pypy_dijkstra(call_params):
    
    call_params = json.loads(call_params)
    print (call_params)
    
    process = subprocess.Popen([str(r) for r in call_params],shell=True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    
    for line in process.stdout: 
        print(line.decode(), end='')
        
    process.stdout.close()
    return_code = process.wait()
    
    return True


def dijkstra_post_oil(df_nodes):
    #### first: generate an adjacent costs matrix for oil.
    logger = logging.getLogger('Post_oil')
    
    # load all the cost pickles
    logger.info('Load all the cost pickles')
    picklefiles = glob.glob(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths','oil','costs','*.pkl'))
    
    pkl_data = {int(os.path.splitext(f.split('/')[-1])[0]):pickle.load(open(f,'rb')) for f in picklefiles}
    print (pkl_data.keys())

    
    df_nodes = df_nodes.reset_index().rename(columns={'index':'idx'})
    target_nodes = df_nodes.loc[df_nodes['NODETYPE'].isin(['CITY','POWERSTATION']),'idx'].values.tolist()
    source_nodes = df_nodes.loc[df_nodes['NODETYPE'].isin(['OILWELL','OILFIELD']) & df_nodes['idx'].isin(pkl_data.keys()),'idx'].values.tolist()
    
    
    pkl_data = {(kk1,kk2):cost for kk1,vv in pkl_data.items() for kk2, cost in vv.items() }
    #print (pkl_data.keys())

    start_nodes = list(set([kk[0] for kk in list(pkl_data.keys()) if kk[0] in source_nodes]))

    
    
    # make them into an nx digraph
    G = nx.DiGraph()
    G.add_edges_from([(kk_tup[0],kk_tup[1], {'z':cost}) for kk_tup, cost in pkl_data.items()])
    
    #print('nodes')
    #print(G.nodes)
    
    # get supply and target nodes on graph, and call dijkstra for all supply nodes
    dijkstra_results = {}
    for sn in tqdm(start_nodes):
        costs,paths = nx.single_source_dijkstra(G, sn, target=None, cutoff=None, weight='z')
        #print (sn)
        #print (costs, paths)
        dijkstra_results[sn] = {'costs':costs,'paths':paths}
    
    # parse results into adjacency matrix
    
    df_adj = pd.DataFrame(
        {sn:{tn:dijkstra_results[sn]['costs'][tn] for tn in target_nodes if tn in dijkstra_results[sn]['costs'].keys()} 
         for sn in start_nodes}) 
    
    idx_mapper = {r[0]:r[1] for r in df_nodes[['idx','NODE']].values.tolist()}
    
    df_adj.index = df_adj.index.astype(int).map(idx_mapper)
    df_adj.columns = df_adj.columns.astype(int).map(idx_mapper)
    
    print (df_adj)
    
    ### second: combine paths
    
    for sn in start_nodes:
        crude_paths = pickle.load(open(f'/paths/oil/{sn}.pkl','rb'))
        
        for tn in dijkstra_results[sn].keys():
            
            master_path = dijkstra_results[sn]['paths'][tn]
            inter_idx = master_path[1]
            
            product_paths = pickle.load(open(f'/paths/oil/{inter_idx}.pkl','rb'))
            
            print ('crude path')
            print (crude_paths[inter_idx])
            print ('product_path')
            print (product_paths[tn])
            
            exit()
            
            
    
    # for each source:
    ## for each target:
    ### combine paths into new dict
    
    # return adjacency matrix
    
    
    
def dijkstra_post_parse(community_nodes):
    if 'COALMINE' in community_nodes['NODETYPE'].unique():
        carrier = 'coal'
        source_types = ['COALMINE']
    elif 'LNGTERMINAL' in community_nodes['NODETYPE'].unique():
        carrier= 'gas'
        source_types = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        source_types = ['OILFIELD','OILWELL']
    
    logger=logging.getLogger(f'{carrier} parse dijkstra')
    community_nodes = community_nodes.reset_index().rename(columns={'index':'idx'})
    
    cost_pkl_fs = glob.glob(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'costs','*'))
    
    logger.info(f'found {len(cost_pkl_fs)} pickle files')
    logger.info('Loading pickles...')
    cost_pkls = {os.path.splitext(os.path.split(el)[1])[0]:pickle.load(open(el,'rb')) for el in cost_pkl_fs}
    
    
    logger.info('Parsing to df')
    df = pd.DataFrame()
    for ii_k, kk in enumerate(list(cost_pkls.keys())):
        if ii_k % 50 ==0:
            print (ii_k)
        new_df = pd.DataFrame(cost_pkls[kk], index=[kk])

        df = df.append(new_df)
    
    idx_mapper = {r[0]:r[1] for r in community_nodes[['idx','NODE']].values.tolist()}
    
    df.index = df.index.astype(int).map(idx_mapper)
    df.columns = df.columns.astype(int).map(idx_mapper)
    df = df.T
    
    return df
    
def paths2h5py():
    ## turn those paths at /paths into h5py
    pass

def dijkstra_post_flow_oil():
    pass
    

def dijkstra_post_flow(node_df, edge_df, params, dijkstra_min_adj):
    ## turn the dijkstra results into flow dfs
    if 'COALMINE' in node_df['NODETYPE'].unique():
        carrier = 'coal'
        source_types = ['COALMINE']
    elif 'LNGTERMINAL' in node_df['NODETYPE'].unique():
        carrier= 'gas'
        source_types = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        source_types = ['OILFIELD','OILWELL']
    
    logger = logging.getLogger(f'Dijkstra_post_{carrier}')
    
    logger.info('loading run data')
    run_data = pickle.load(open(params['flowfill_run'][carrier],'rb'))  
    SCALE_FACTOR = run_data['SCALE_FACTOR']
    STEP_INI = run_data['STEP_INI']
    GAMMA = run_data['GAMMA']
    df_alpha = run_data['ALPHA']
    
    logger.info('running fill algo')
    ii_w, df_flow, df_z = differential_fill(node_df, dijkstra_min_adj, df_alpha, STEP_INI, GAMMA,SCALE_FACTOR, params, logging.getLogger('ini_diff_fill'), None)
    
    
    logger.info('prepping dfs')
    sources = df_flow.sum()[df_flow.sum()>0].index.tolist()
    
    node_df = node_df.reset_index().rename(columns={'index':'idx'}).set_index('NODE')
    
    edge_df['flow']=0
    edge_df = pd.merge(edge_df, node_df[['idx']], how='left',left_on='source',right_index=True).rename(columns={'idx':'source_idx'})
    edge_df = pd.merge(edge_df, node_df[['idx']], how='left',left_on='target',right_index=True).rename(columns={'idx':'target_idx'})
    edge_df = edge_df.set_index(['source_idx','target_idx'])
    
    logger.info('filling flow paths')
    for source in tqdm(sources, desc='adding flow'):
        source_idx = node_df.at[source,'idx']
        paths = pickle.load(open(f'/paths/{carrier}/{source_idx}.pkl','rb'))
        for dest in df_flow.loc[df_flow[source]>0,source].index.values.tolist():
            dest_idx = node_df.at[dest,'idx']
            add_flow = df_flow.at[dest,source]
            node_path_list = paths[dest_idx]
            node_path_tups = list(zip(node_path_list[0:-1],node_path_list[1:]))
            edge_df.loc[node_path_tups,'flow']+=add_flow

    return edge_df.reset_index().drop(columns=['source_idx','target_idx'])
    

        
def dijkstra_pypy_paths(community_nodes, params):
    if 'COALMINE' in community_nodes['NODETYPE'].unique():
        carrier = 'coal'
        source_types = ['COALMINE']
    elif 'LNGTERMINAL' in community_nodes['NODETYPE'].unique():
        carrier= 'gas'
        source_types = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        source_types = ['OILFIELD','OILWELL']
        
    logger=logging.getLogger('calling dijkstra paths')
    logger.info(f'doing carrier {carrier}')
    
    logger.info('Getting supply nodes')
    
    community_nodes = community_nodes.reset_index().rename(columns={'index':'idx'})
    #supply_nodes = community_nodes.loc[community_nodes['NODETYPE'].isin(source_types),'idx'].astype(int).values.tolist()
    supply_nodes = pickle.load(open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'supply_B_nodes.pkl'),'rb'))

    
    
    logger.info('calling mp')
    executable = 'bin/pypy/pypy3.7-v7.3.3-linux64/bin/pypy'
    script = 'flow_fill_dijkstra.py'
    edges_file = os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'edges_B.pkl')
    nodes_file = os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'supply_B_nodes.pkl')
    target_nodes = os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'target_B_nodes.pkl')
    
    #call_params = [json.dumps([executable, script, edges_file, nodes_file, target_nodes, CHUNK*ii, CHUNK*(ii+1)])
    #               for ii in range(N_WORKERS)]
    
    """ # Normal call params
       
    CHUNK = len(supply_nodes)//N_WORKERS +1
    call_params = [[executable, script, edges_file, nodes_file, target_nodes, str(CHUNK*ii), str(CHUNK*(ii+1)), str(ii)]
                   for ii in range(N_WORKERS)]
    """
    
    
    cost_pkls = glob.glob(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'costs','*.pkl'))
    cost_pkl_ints = [int(os.path.splitext(el.split('/')[-1])[0]) for el in cost_pkls]
    
    # check workers
    """
    
    
    for call_param in call_params[0:2]:
        start_idx = int(call_param[5])
        end_idx = int(call_param[6])
        ii_w = int(call_param[7])
        
        for sn in supply_nodes[start_idx:end_idx]:
            print (f'Worker: {ii_w}\t sn: {sn}\t done: {sn in cost_pkl_ints}')     
    """
    
    missing_sn = [sn for sn in supply_nodes if sn not in cost_pkl_ints]
    logger.info(f'N missing sn: {len(missing_sn)}')
    
    logger.info('writing new worker nodes and calling executable')
    # new worker params
    CHUNK = len(missing_sn)//N_WORKERS +1
    
    for ii in range(N_WORKERS):
        pickle.dump(missing_sn[CHUNK*ii:CHUNK*(ii+1)], open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,str(ii)+'_workernodes.pkl'),'wb'))
        
    call_params = [
        [
            executable, 
            script, 
            edges_file, 
            os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,str(ii)+'_workernodes.pkl'), 
            target_nodes, 
            'nan', 
            'nan', 
            str(ii)
        ] for ii in range(N_WORKERS)
    ]
        
    
    processes = {}
    for worker in range(N_WORKERS):
        print ('calling worker', worker)
        processes[worker] = subprocess.Popen(call_params[worker], stdout = subprocess.PIPE, stderr=subprocess.PIPE)
        
    print ('processes',processes)
    print ('sleeping')
    time.sleep(10)
    print ('processes',processes, [pp.poll() for pp in processes.values()])
    
        
    while True:
        
        for worker, process in processes.items():
            # check for output
            for line in process.stdout:
                print(line.decode(), end='')
        
        # poll for completion
        if not np.any([process.poll()==None for worker, process in processes.items()]):
            break
            
        time.sleep(5)
    
    
    #pool = mp.Pool(N_WORKERS)
    #results = pool.map(call_pypy_dijkstra, call_params)
    
    #print ('results',results)
        
    return []

def genetic_wrapper(community_nodes, dijkstra_mincost_adj, iso2, node_iso2, df_trade, df_energy, params):
    
    # logic -> high-production countries should have lower cost for production
    
    # loss -> production difference + arc difference
    do_='ini'
    DROP_CA=True
    
    
    if 'COALMINE' in community_nodes['NODETYPE'].unique():
        carrier = 'coal'
        energy_col = 'Coal*'
        trade_col = 'Qty'
        source_types = ['COALMINE']
    elif 'LNGTERMINAL' in community_nodes['NODETYPE'].unique():
        carrier= 'gas'
        energy_col = 'Natural gas'
        source_types = ['OILFIELD','OILWELL']
    else:
        carrier='oil'
        energy_col = 'Crude oil'
        source_types = ['OILFIELD','OILWELL']
    
    logger = logging.getLogger(f'{carrier}_FitOpt')
    
    # prep trade
    df_trade = df_trade.loc[(~df_trade['Partner ISO'].isna())&(~df_trade['Reporter ISO'].isna())&(df_trade['Year']==2018)&(df_trade['Trade Flow']=='Import')&(df_trade['Reporter ISO']!='WLD')&(df_trade['Partner ISO']!='WLD'),['Reporter ISO','Partner ISO','Qty']]
    
    print('df_trade')
    print (df_trade)
    
    iso2.loc[iso2['iso2']=="'NA",'iso2']='NA'
    
    df_trade = pd.merge(df_trade,iso2[['iso2','iso3']], how='left',left_on='Reporter ISO', right_on='iso3').drop(columns=['iso3']).rename(columns={'iso2':'reporter_iso2'})
    df_trade = pd.merge(df_trade,iso2[['iso2','iso3']], how='left',left_on='Partner ISO', right_on='iso3').drop(columns=['iso3']).rename(columns={'iso2':'partner_iso2'})
    
    # trade_qty in kg, convert to TJ
    df_trade['TJ'] = df_trade['Qty']/1000/params['tperTJ'][carrier]
    df_energy[f'{carrier}_TJ'] = df_energy[energy_col]*41868/1000 #ktoe-> TJ
    
    print ('df trade2')
    print (df_trade)
    df_trade['reporter_iso2'] = df_trade['reporter_iso2'].str.replace("'NA",'NA')
    df_trade['partner_iso2'] = df_trade['partner_iso2'].str.replace("'NA",'NA')
    df_trade = df_trade[['reporter_iso2','partner_iso2','TJ']] \
                    .groupby(['reporter_iso2','partner_iso2']) \
                    .sum() \
                    .unstack() \
                    .rename_axis(['meow','partner_iso2'],axis='columns') \
                    .droplevel('meow', axis=1) \
    
    
    # clean up energy
    df_energy = df_energy[~df_energy['ISO_A2'].isin(['lo','WORLD'])]
    df_energy['ISO_A2'] = df_energy['ISO_A2'].astype(str)
    df_energy['ISO_A2'] = df_energy['ISO_A2'].str.replace("nan",'NA')
    df_energy['ISO_A22'] = df_energy['ISO_A2']
    df_energy = df_energy[['ISO_A2','ISO_A22',f'{carrier}_TJ']].set_index(['ISO_A2','ISO_A22']).unstack().rename_axis('reporter_iso2').rename_axis(['meow','partner_iso2'],axis='columns').droplevel('meow', axis=1) 
    
    ann_prod = pd.Series(np.diag(df_energy), index=[df_energy.index], name='TJ')
    ann_prod.index = ann_prod.index.get_level_values(0)
    
    #######
    ### pick initial guess - scale by 
    #######

    
    community_nodes = pd.merge(community_nodes, node_iso2, how='left',left_on='NODE',right_on='NODE')
    df_alpha = community_nodes.loc[community_nodes['NODETYPE'].isin(source_types),['NODE','iso2']].set_index('NODE')
    
    if DROP_CA==True:
        print ('drop_ca')
        drop_ca =json.load(open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','drop_CA_coalmines.json'),'r'))
        df_alpha = df_alpha[~df_alpha.index.isin(drop_ca)]
        print (df_alpha)
        
    
    
    # handle missing countries
    df_por = ann_prod.loc[ann_prod.index.isin(df_alpha['iso2'].unique().tolist())] / ann_prod.loc[ann_prod.index.isin(df_alpha['iso2'].unique().tolist()+['REM'])].sum() /df_alpha.groupby('iso2').size()
    nan_iso2 = df_por[df_por.isna()].index.values.tolist()
    N_iso2 = df_alpha.groupby('iso2').size().loc[nan_iso2].sum()
    df_por.loc[df_por.isna()] = (ann_prod.loc[ann_prod.index=='REM'] / ann_prod.loc[ann_prod.index.isin(df_alpha['iso2'].unique().tolist()+['REM'])].sum() / N_iso2).values[0]
    df_por.name = 'portion'

    df_alpha = pd.merge(df_alpha, df_por, how='left',left_on='iso2',right_index=True).rename(columns={'portion':'alpha'})
    df_alpha = df_alpha['alpha']
    df_alpha.loc[df_alpha==0] = df_alpha[df_alpha>0].min() # bump up to min>0
    df_alpha = 1/df_alpha # inv so small portion is large cost increase
    df_alpha = df_alpha / df_alpha.max() # norm
    
    df_alpha_ini = df_alpha.copy()
    
    """
    def chroma2params(chroma, df_alpha_ini):
        df = pd.Series(10**(np.array(chroma[1:])*3-3), index=df_alpha_ini.index)
        df.name='alpha'
        SF = 10**(2+chroma[0]*4)
        return SF, df
    """
    
    def chroma2params(chroma, df_alpha_ini):
        df = pd.Series(10**(np.array(chroma)*3-3), index=df_alpha_ini.index)
        df.name='alpha'
        SF = 10 # **(2+chroma[0]*4)
        return SF, df
        
    def params2chroma(SF, df_alpha):
        #return [(np.log10(SF)-2)/4] + ((np.log10(df_alpha.values)+3)/3).tolist()
        return ((np.log10(df_alpha.values)+3)/3).tolist()
        
    def flow2iso2adj(df_flow, node_iso2):
        return pd.merge(
                    pd.merge(
                        df_flow, 
                        node_iso2[['NODE','iso2']], 
                        how='left',
                        left_index=True, 
                        right_on='NODE'
                    ).drop(columns=['NODE']).groupby('iso2').sum().T.reset_index(), 
                     node_iso2[['NODE','iso2']],
                     how='left',
                     left_on='index',
                     right_on='NODE'
                ).drop(columns=['NODE','index']).groupby('iso2').sum()

    
    if do_=='ini':
    
        SCALE_FACTOR = 10
        STEP_INI = 15
        GAMMA = 0.8

        ii_w, df_flow, df_z = differential_fill(community_nodes, dijkstra_mincost_adj, df_alpha, STEP_INI, GAMMA,SCALE_FACTOR, params, logging.getLogger('ini_diff_fill'), None)
        
        print (df_flow)
        
        pickle.dump(df_flow, open('./tmp_df_flow.pkl','wb'))

        SIM_ADJ = flow2iso2adj(df_flow, node_iso2)

        #SIM_ADJ.to_csv('./sim_adj.csv')
        
        weighted_prod_loss, weighted_trade_loss = get_loss(SIM_ADJ, ann_prod, df_trade)
        
        record = {
            'SCALE_FACTOR':SCALE_FACTOR,
            'STEP_INI':STEP_INI,
            'GAMMA':GAMMA,
            'SIM_ADJ':SIM_ADJ,
            'ALPHA':df_alpha,
            'weighted_prod_loss':weighted_prod_loss,
            'weighted_trade_loss':weighted_trade_loss,
        }
        
        print (SIM_ADJ)
        print (f'SCALE_FACTOR: {SCALE_FACTOR}, prod_loss: {weighted_prod_loss:.2f}, trade_loss: {weighted_trade_loss:.2f}')
        
        pickle.dump(record, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow',carrier,'genetic','0.pkl'),'wb'))
        
    if do_=='ini_batch':
        offset=1
        
        
        N_WORKERS = 8
        
        #SCALE_FACTOR = 100000
        STEP_INI = 15
        GAMMA = 0.8
        
        differential_fill_params = [
            [
                community_nodes,
                dijkstra_mincost_adj,  
                df_alpha * (1+(np.random.rand(len(df_alpha))-0.5)/5), #+- 10%
                STEP_INI,
                GAMMA,
                10**(2+np.random.rand()*4), # try some random scaling factors between 10**2 and 10**6
                params,
                logging.getLogger(f'dijkstra_fill_worker_{ii_w}'),
                ii_w,
            ]
            for ii_w in range(N_WORKERS)
        ]
        
        
        pool = mp.Pool(N_WORKERS)
        
        results = pool.starmap(differential_fill, differential_fill_params)
        
        for ii_w, df_flow, df_z in results:
            SIM_ADJ = flow2iso2adj(df_flow, node_iso2)
            
            weighted_prod_loss, weighted_trade_loss = get_loss(SIM_ADJ, ann_prod, df_trade)

            record = {
                'SCALE_FACTOR':differential_fill_params[ii_w][5],
                'STEP_INI':STEP_INI,
                'GAMMA':GAMMA,
                'SIM_ADJ':SIM_ADJ,
                'ALPHA':differential_fill_params[ii_w][2],
                'weighted_prod_loss':weighted_prod_loss,
                'weighted_trade_loss':weighted_trade_loss,
            }
            
            #print (SIM_ADJ)
            print ('results:',ii_w, rec['SCALE_FACTOR'], weighted_prod_loss, weighted_trade_loss)

            pickle.dump(record, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow',carrier,f'{offset+ii_w}.pkl'),'wb'))
            
    if do_=='run_genetic':
        # start with a good baseline and then mutant some small genes to try to get better.
        #res_init = pickle.load(open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','coal','0.pkl'),'rb'))


        
        #run_files = glob.glob(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','coal','*.pkl'))
        #run_files = glob.glob(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','coal','genetic','*.pkl'))
        run_files = glob.glob(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow',carrier,'exp2','*.pkl'))
        records = []
        for f in run_files:
            records.append(pickle.load(open(f,'rb')))
        
        
        offset = max([int(os.path.split(os.path.splitext(f)[0])[1]) for f in run_files if not 'ini' in f])
        print ('offset:',offset)

        N_WORKERS = 10
        STEP_INI = 15
        GAMMA = 0.8
        pool = mp.Pool(N_WORKERS)
        w_prod = 0.75
        w_trad = 0.25
        MUTATION_RATE=0.15
        N_ITERS=50
        df_alpha_ini = df_alpha.copy()
        
        print ('df_alpha_ini')
        print (df_alpha_ini)
        
        for r in records:
            r['fitness'] = 0.75*r["weighted_prod_loss"] + 0.25*r["weighted_trade_loss"]
            
        rec_init = sorted(records, key=lambda r: r['fitness'])[0]

        best_chromasome = params2chroma(rec_init['SCALE_FACTOR'], rec_init['ALPHA'])
        best_fitness = w_prod*rec_init["weighted_prod_loss"] + w_trad*rec_init["weighted_trade_loss"]

        print(f'ini SF {rec_init["SCALE_FACTOR"]} best fitness {best_fitness}')


        def cross_over(chromasome_a, chromasome_b):

            chromasome_b = np.array(chromasome_b)
            chromasome_a = np.array(chromasome_a)
            children = []

            for ii_w in range(N_WORKERS):

                idx_swap = np.random.choice(len(chromasome_a), len(chromasome_a)//2)

                new_chroma = chromasome_a.copy()
                new_chroma[idx_swap] = chromasome_b[idx_swap]

                children.append(new_chroma.tolist())

            return children


        def mutate(population,MUTATION_RATE):
            pop_arr = np.array(population)
            mult = np.ones(pop_arr.shape)
            aug = np.random.rand(*pop_arr.shape) 
            mask = np.random.rand(*pop_arr.shape)<=MUTATION_RATE
            mult[mask] += aug[mask]
            pop_arr *= mult

            return pop_arr.tolist()


        def run_population(population, offset):


            differential_fill_params = []
            for ii_w, chroma in enumerate(population):
                SF, df = chroma2params(chroma, df_alpha_ini)
                #print ('df')
                #print (df)

                differential_fill_params.append(
                    [
                        community_nodes,
                        dijkstra_mincost_adj,  
                        df,
                        STEP_INI,
                        GAMMA,
                        SF, # try some random scaling factors between 10**2 and 10**6
                        params,
                        logging.getLogger(f'dijkstra_fill_worker_{ii_w}'),
                        ii_w,
                    ]
                )


            results = pool.starmap(differential_fill, differential_fill_params)

            fitness = {}
            for ii_w, df_flow, df_z in results:
                SIM_ADJ = flow2iso2adj(df_flow, node_iso2)

                weighted_prod_loss, weighted_trade_loss = get_loss(SIM_ADJ, ann_prod, df_trade)

                record = {
                    'SCALE_FACTOR':differential_fill_params[ii_w][5],
                    'STEP_INI':STEP_INI,
                    'GAMMA':GAMMA,
                    'SIM_ADJ':SIM_ADJ,
                    'ALPHA':differential_fill_params[ii_w][2],
                    'weighted_prod_loss':weighted_prod_loss,
                    'weighted_trade_loss':weighted_trade_loss,
                }
                fitness[ii_w] = w_prod*record["weighted_prod_loss"] + w_trad*record["weighted_trade_loss"]

                #print (SIM_ADJ)
                print ('results:',ii_w,record['SCALE_FACTOR'], weighted_prod_loss, weighted_trade_loss)

                # pickle.dump(record, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','coal','genetic',f'{offset+ii_w}.pkl'),'wb'))
                pickle.dump(record, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow',carrier,'exp2',f'{offset+ii_w}.pkl'),'wb'))


            return fitness




        for ITER in range(N_ITERS):

            if ITER==0:
                # on first iter, just mutate the best chromasome
                logger.info('first iteration, generate mutation')
                population = mutate([best_chromasome]*N_WORKERS, 0.15)

                # get fitness
                logger.info('Running first populaion')
                fitness = run_population(population, offset)

            else:
                # sort the chromasomes by fitness
                print ('fitness')
                print (fitness)
                sort_fitness = {k: v for k, v in sorted(fitness.items(), key=lambda item: item[1]) if v<best_fitness}
                print ('sort_fitenss')
                print (sort_fitness)

                # if >=2 are improvements, take crossover the two best then mutate all the chromasomes
                if len(sort_fitness)>=2:
                    print ('multi improved')
                    population = cross_over(population[list(sort_fitness.keys())[0]], population[list(sort_fitness.keys())[1]])
                    population = mutate(population,MUTATION_RATE)
                    best_chromasome = population[list(sort_fitness.keys())[0]]
                    best_fitness = sort_fitness[list(sort_fitness.keys())[0]]

                elif len(sort_fitness)==1:
                    print ('single improved')
                    population = cross_over(population[list(sort_fitness.keys())[0]], best_chromasome)
                    population = mutate(population,MUTATION_RATE)
                    best_chromasome = population[list(sort_fitness.keys())[0]]
                    best_fitness = sort_fitness[list(sort_fitness.keys())[0]]

                else:
                    print ('else remuate')
                    population = mutate([best_chromasome]*N_WORKERS, MUTATION_RATE)

                fitness = run_population(population, offset)
                offset+=N_WORKERS
            

    return []