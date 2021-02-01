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
import h5py
import gc


from ffsc.flow.simplex import network_simplex
from ffsc.interdiction.gp import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

N_WORKERS=2

# total cost overflowing max int: 2147483647

import multiprocessing as mp

def dijkstra_pypy_pickle(community_nodes, df_edges, params):
    """
    Pickle out supply nodes, target nodes, and edges
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
        intermediate_types = ['REFINERY']
        
        
    logger = logging.getLogger(f'{carrier} pypy pickle')
    if carrier == 'gas':
        keep_supply = json.load(open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','keep_oilfields_gas.json'),'r')) \
                    + json.load(open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','keep_oilwells_gas.json'),'r'))
        logger.info(f'Got Keepsupply len {len(keep_supply)}')
    elif carrier=='oil':
        keep_supply = json.load(open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','keep_oil.json'),'r'))
        keep_refineries = json.load(open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','keep_refineries.json'),'r'))
    else:
        # carrier=='coal':
        drop_CA = json.load(open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','drop_CA_coalmines.json'),'r'))
        keep_supply = community_nodes.loc[(community_nodes['NODETYPE'].isin(source_types)) & (~community_nodes['NODE'].isin(drop_CA)),'NODE'].values.tolist()
        
    
    logger.info(f'Got Keepsupply len {len(keep_supply)}')
    
    
    logger.info('prepping dfs')
    
    df_edges = df_edges[df_edges['source']!='supersource']
    community_nodes = community_nodes.reset_index().rename(columns={'index':'idx'})
    community_nodes['idx'] = community_nodes['idx'].astype(int)
    df_edges = pd.merge(df_edges, community_nodes[['idx','NODE','NODETYPE']], how='left',left_on='source',right_on='NODE').drop(columns=['NODE']).rename(columns={'idx':'source_idx','NODETYPE':'source_type'})
    df_edges = pd.merge(df_edges, community_nodes[['idx','NODE','NODETYPE']], how='left',left_on='target',right_on='NODE').drop(columns=['NODE']).rename(columns={'idx':'target_idx','NODETYPE':'target_type'})

    #print (df_edges)
    target_types = ['CITY','POWERSTATION']

    if carrier in ['gas','coal']:
        logger.info('prepping graph')
        G = nx.DiGraph()
        G.add_edges_from([(r[0],r[1], {'z':r[2]}) for r in df_edges[['source_idx','target_idx','z']].astype(int).values.tolist()])

        supply_nodes = community_nodes.loc[(community_nodes['NODETYPE'].isin(source_types)) & (community_nodes['NODE'].isin(keep_supply)),'idx'].astype(int).values.tolist()
        target_nodes = community_nodes.loc[community_nodes['NODETYPE'].isin(target_types),'idx'].astype(int).values.tolist()

        logger.info('pickling')
        pickle.dump(supply_nodes, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'supply_nodes.pkl'),'wb'),protocol=2)
        pickle.dump(target_nodes, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'target_nodes.pkl'),'wb'),protocol=2)
        pickle.dump(G._succ,open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'edges.pkl'),'wb'),protocol=2)
    
    else: # is oil and is tricky.
        logger.info('Prepping crude/products distinction')
        
        # make lastmile cities-ports is on the 'B' side
        df_edges.loc[(df_edges['target_type']=='CITY')&(df_edges['source_type']=='PORT'),'source'] = df_edges.loc[(df_edges['target_type']=='CITY')&(df_edges['source_type']=='PORT'),'source'].apply(lambda el: el+'_B')
        df_edges.loc[(df_edges['source_type']=='CITY')&(df_edges['target_type']=='PORT'),'target'] = df_edges.loc[(df_edges['source_type']=='CITY')&(df_edges['target_type']=='PORT'),'target'].apply(lambda el: el+'_B')
        df_edges['source_B'] = df_edges['source'].apply(lambda el: el[-2:]=='_B')
        df_edges['target_B'] = df_edges['target'].apply(lambda el: el[-2:]=='_B')
        df_edges['side']='A'
        cond_B = df_edges['source_type'].isin(['CITY','POWERSTATION']) | df_edges['target_type'].isin(['CITY','POWERSTATION']) | (df_edges['source_B']==True) | (df_edges['target_B']==True)
        df_edges.loc[cond_B,'side']='B'
        
        logger.info('Prepping crude graph')
        G = nx.DiGraph()
        G.add_edges_from([(r[0],r[1], {'z':r[2]}) for r in df_edges.loc[df_edges['side']=='A',['source_idx','target_idx','z']].astype(int).values.tolist()])

        supply_nodes = community_nodes.loc[(community_nodes['NODETYPE'].isin(source_types)) & (community_nodes['NODE'].isin(keep_supply)),'idx'].astype(int).values.tolist()
        target_nodes = community_nodes.loc[community_nodes['NODETYPE'].isin(intermediate_types),'idx'].astype(int).values.tolist()

        logger.info('pickling crude')
        pickle.dump(supply_nodes, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'supply_A_nodes.pkl'),'wb'),protocol=2)
        pickle.dump(target_nodes, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'target_A_nodes.pkl'),'wb'),protocol=2)
        pickle.dump(G._succ,open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'edges_A.pkl'),'wb'),protocol=2)
        
        logger.info('Prepping products graph')
        G = nx.DiGraph()
        G.add_edges_from([(r[0],r[1], {'z':r[2]}) for r in df_edges.loc[df_edges['side']=='B',['source_idx','target_idx','z']].astype(int).values.tolist()])

        supply_nodes = community_nodes.loc[(community_nodes['NODE'].isin(keep_refineries)),'idx'].astype(int).values.tolist()
        target_nodes = community_nodes.loc[community_nodes['NODETYPE'].isin(target_types),'idx'].astype(int).values.tolist()

        logger.info('pickling products')
        pickle.dump(supply_nodes, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'supply_B_nodes.pkl'),'wb'),protocol=2)
        pickle.dump(target_nodes, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'target_B_nodes.pkl'),'wb'),protocol=2)
        pickle.dump(G._succ,open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_paths',carrier,'edges_B.pkl'),'wb'),protocol=2)
        
    
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
    inter_nodes = df_nodes.loc[df_nodes['NODETYPE'].isin(['REFINERY']),'idx'].values.tolist()
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
    for sn in tqdm(start_nodes, desc='shortest dijkstra'):
        costs,paths = nx.single_source_dijkstra(G, sn, target=None, cutoff=None, weight='z')
        #print (sn)
        #print (costs, paths)
        dijkstra_results[sn] = {'costs':costs,'paths':paths}
    
    # parse results into adjacency matrix
    
    df_adj = pd.DataFrame(
        {sn:{tn:dijkstra_results[sn]['costs'][tn] for tn in target_nodes if tn in dijkstra_results[sn]['costs'].keys()} 
         for sn in start_nodes}) 
    

    """
    path_files = glob.glob('/paths/oil/*.pkl')
    path_files = [f for f in path_files if int(os.path.splitext(f.split('/')[-1])[0]) in inter_nodes]
    
    for ii_f, f in enumerate(path_files):
        if ii_f>139:
            idx = int(os.path.splitext(f.split('/')[-1])[0])
            data = pickle.load(open(f,'rb'))

            for kk,vv in tqdm(data.items(), desc=f'{ii_f}/{len(path_files)}'):
                if not os.path.exists(f'/paths/oil_json/{idx}-{kk}.json'):
                    json.dump(vv, open(f'/paths/oil_json/{idx}-{kk}.json','w'))

            gc.collect()
            
    """
    
            

    
            
        
    #    if idx in inter_nodes:
    #        inter_paths[idx] = pickle.load(open(f,'rb'))
    
    ### second: combine paths
    logger.info('combining paths')
    for ii_s, sn in enumerate(start_nodes):
        crude_paths = pickle.load(open(f'/paths/oil/{sn}.pkl','rb'))
        full_paths = {}
        
        for tn in tqdm(target_nodes, desc=f'{ii_s}/{len(start_nodes)}'):
            if tn in dijkstra_results[sn]['paths'].keys():
            
                #print (dijkstra_results[sn]['paths'])

                master_path = dijkstra_results[sn]['paths'][tn]

                inter_idx = master_path[1]

                #full_paths[tn] = crude_paths[inter_idx] + h5_objs[inter_idx][str(tn)][1:].values.tolist()
                
                full_paths[tn] = crude_paths[inter_idx] + json.load(open(f'/paths/oil_json/{inter_idx}-{tn}.json','r'))
                
        pickle.dump(full_paths, open(f'/paths/oil_join/{sn}.pkl','wb'))
                
                
            
    idx_mapper = {r[0]:r[1] for r in df_nodes[['idx','NODE']].values.tolist()}
            
    df_adj.index = df_adj.index.astype(int).map(idx_mapper)
    df_adj.columns = df_adj.columns.astype(int).map(idx_mapper)
    print (df_adj)
    
    # for each source:
    ## for each target:
    ### combine paths into new dict
    
    return df_adj
    
    
    
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
    #ii_w, df_flow, df_z = differential_fill(node_df, dijkstra_min_adj, df_alpha, STEP_INI, GAMMA,SCALE_FACTOR, params, logging.getLogger('ini_diff_fill'), None)
    
    #pickle.dump(df_flow, open('./tmp_df_flow.pkl','wb'))
    df_flow = pickle.load(open('./tmp_df_flow.pkl','rb'))
    
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
        if carrier=='oil':
            paths = pickle.load(open(f'/paths/{carrier}_join/{source_idx}.pkl','rb'))
        else:
            paths = pickle.load(open(f'/paths/{carrier}/{source_idx}.pkl','rb'))
        for dest in df_flow.loc[df_flow[source]>0,source].index.values.tolist():
            dest_idx = node_df.at[dest,'idx']
            add_flow = df_flow.at[dest,source]
            node_path_list = paths[dest_idx]
            node_path_tups = list(zip(node_path_list[0:-1],node_path_list[1:]))
            if carrier=='oil':
                # make sure the idxs arent the same
                node_path_tups = [nn for nn in node_path_tups if nn[0]!=nn[1]]
            edge_df.loc[node_path_tups,'flow']+=add_flow
            
        gc.collect()

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
    
def get_loss(SIM_ADJ, ann_prod, df_trade):
    
    # rebase units
    SIM_ADJ = SIM_ADJ*ann_prod[ann_prod.index.isin(SIM_ADJ.sum(axis=1).index.values)].sum()/SIM_ADJ.sum().sum()
    
    # annual production
    sim_prod = SIM_ADJ.sum(axis=1)
    sim_prod.name='sim_prod'
    prod_loss_df = pd.merge(ann_prod, sim_prod, how='left', left_index=True, right_index=True).sort_values('TJ').rename(columns={'TJ':'actual','sim_prod':'simulation'})
    prod_loss_df = prod_loss_df.loc[~prod_loss_df['simulation'].isna(),:]
    prod_loss_df['per_diff'] = (prod_loss_df['simulation'] - prod_loss_df['actual'])/prod_loss_df['actual']
    weighted_prod_loss = (prod_loss_df['per_diff'].abs()*prod_loss_df['actual']/prod_loss_df['actual'].sum()).sum()
    
    # do trade pairs
    # reporter -> importer; partner -> producer
    actual_pairs = df_trade.unstack()
    actual_pairs.name='actual'
    sim_pairs = SIM_ADJ.unstack().rename_axis(['partner_iso2','reporter_iso2'])
    sim_pairs.name = 'simulation'
    all_pairs = pd.merge(actual_pairs, sim_pairs, how='outer',left_index=True, right_index=True).reset_index().rename(columns={'partner_iso2':'source','reporter_iso2':'target'})
    all_pairs = all_pairs[~(all_pairs['actual'].isna())].fillna(0)
    all_pairs = all_pairs[~(all_pairs['source']==all_pairs['target'])].sort_values('actual')
    all_pairs['per_diff'] = (all_pairs['simulation'] - all_pairs['actual'])/all_pairs['actual']
    weighted_trade_loss = (all_pairs['per_diff'].abs()*all_pairs['actual']/all_pairs['actual'].sum()).sum()
    
    return weighted_prod_loss, weighted_trade_loss

def differential_fill(community_nodes, df_z_ini, supply_alpha,STEP_INI, GAMMA,SCALE_FACTOR, params, logger=None, ii_w=None):
    """
    supply_alpha: {source:float}
    GAMMA <1 -> slow step increase; GAMMA >1 -> speed up increase
    differential fill algo, marginal supply cost of the form: alpha*flow + dist
    (supply cost: alpha/2*flow**2 + dist*flow)
    dijkstra pairs: {source:{target:{dist:shortest_z, path:path} for target in reachable_targets} for source in sources}
    """    
    
    # set up dfs - source is column, target is index
    
    #df_z_ini = pd.DataFrame({source:{target:vv2['dist'] for target,vv2 in vv1.items()} for source,vv1 in disjkstra_pairs})
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
        
    if not logger:
        logger = logging.getLogger(f'Dijkstra shortest {carrier}')
    
    comm_col = f'comm_{params["COMMUNITY_LEVEL"][carrier]}'
    
    df_dem = pd.Series({r[0]:r[1] for r in community_nodes.loc[community_nodes['NODETYPE'].isin(['CITY','POWERSTATION']) & (community_nodes['NODE'].isin(df_z_ini.index)), ['NODE','D']].values.tolist()})  
    
    supply_alpha = supply_alpha*SCALE_FACTOR
    
    ### loop:
    demand_ini = df_dem.sum()
    N_ini = len(df_dem)
    ITER = 10
    STEP = STEP_INI
    ii = 0
    while df_dem.sum()>0: # while any demand is left while df_dem.sum()>0
        tic = time.time()

        # add_flow should be a vector in dimension of the target
        add_flow = pd.Series(STEP, index=df_dem.index).combine(df_dem, min, 0)
        
        #df_z_prime = df_z + supply_alpha * STEP

        # add the flow to the min cost pair
        # df_flow.unstack().loc[[(v,k) for k,v in df_z_prime.T.idxmin().iteritems()]] += add_flow.values
        df_flow.unstack().loc[[(v,k) for k,v in df_z.T.idxmin().iteritems()]] += add_flow.values

        # subtrace the flow from outstanding demand
        df_dem -= add_flow

        N = (df_dem>0).sum()

        # update df_cost      
        df_z = df_z_ini + supply_alpha * df_flow.sum()
        logger.info(f'{ii} \t {STEP} \t {df_dem.sum()} \t {(df_dem>0).sum()} \t {time.time()-tic}')
        
        STEP = int(STEP_INI/max((N/N_ini)**GAMMA,0.001))
        ii+=1
        
    return ii_w, df_flow, df_z
    
"""DEP
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
    
    #try to get shortest path using A*   
    
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
            
            #meant to estimate the distance between the source and the target
            
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

def bayesian_wrapper(community_nodes, dijkstra_mincost_adj, iso2, node_iso2, df_trade, df_energy, params):
    
    # logic -> high-production countries should have lower cost for production
    
    # loss -> production difference + arc difference
    do_='run_genetic'
    DROP_CA=True
    
    logger = logging.getLogger('FitOpt')
    
    
    # prep trade
    df_trade = df_trade.loc[(~df_trade['Partner ISO'].isna())&(~df_trade['Reporter ISO'].isna())&(df_trade['Year']==2018)&(df_trade['Trade Flow']=='Import')&(df_trade['Reporter ISO']!='WLD')&(df_trade['Partner ISO']!='WLD'),['Reporter ISO','Partner ISO','Qty']]
    
    iso2.loc[iso2['iso2']=="'NA",'iso2']='NA'
    
    df_trade = pd.merge(df_trade,iso2[['iso2','iso3']], how='left',left_on='Reporter ISO', right_on='iso3').drop(columns=['iso3']).rename(columns={'iso2':'reporter_iso2'})
    df_trade = pd.merge(df_trade,iso2[['iso2','iso3']], how='left',left_on='Partner ISO', right_on='iso3').drop(columns=['iso3']).rename(columns={'iso2':'partner_iso2'})
    
    # trade_qty in kg, convert to TJ
    df_trade['TJ'] = df_trade['Qty']/1000/params['tperTJ']['coal']
    df_energy['coal_TJ'] = df_energy['Coal*']*41868/1000 #ktoe-> TJ
    
    
    df_trade['reporter_iso2'] = df_trade['reporter_iso2'].str.replace("'NA",'NA')
    df_trade['partner_iso2'] = df_trade['partner_iso2'].str.replace("'NA",'NA')
    df_trade = df_trade[['reporter_iso2','partner_iso2','TJ']].set_index(['reporter_iso2','partner_iso2']).unstack().rename_axis(['meow','partner_iso2'],axis='columns').droplevel('meow', axis=1)
    
    
    # clean up energy
    df_energy = df_energy[~df_energy['ISO_A2'].isin(['lo','WORLD'])]
    df_energy['ISO_A2'] = df_energy['ISO_A2'].astype(str)
    df_energy['ISO_A2'] = df_energy['ISO_A2'].str.replace("nan",'NA')
    df_energy['ISO_A22'] = df_energy['ISO_A2']
    df_energy = df_energy[['ISO_A2','ISO_A22','coal_TJ']].set_index(['ISO_A2','ISO_A22']).unstack().rename_axis('reporter_iso2').rename_axis(['meow','partner_iso2'],axis='columns').droplevel('meow', axis=1) 
    
    ann_prod = pd.Series(np.diag(df_energy), index=[df_energy.index], name='TJ')
    ann_prod.index = ann_prod.index.get_level_values(0)
    
    #######
    ### pick initial guess - scale by 
    #######

    
    community_nodes = pd.merge(community_nodes, node_iso2, how='left',left_on='NODE',right_on='NODE')
    df_alpha = community_nodes.loc[community_nodes['NODETYPE'].isin(['COALMINE']),['NODE','iso2']].set_index('NODE')
    
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

    
    if do_=='ini':
    
        SCALE_FACTOR = 10
        STEP_INI = 15
        GAMMA = 0.8

        ii_w, df_flow, df_z = differential_fill(community_nodes, dijkstra_mincost_adj, df_alpha, STEP_INI, GAMMA,SCALE_FACTOR, params, logging.getLogger('ini_diff_fill'), None)

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
        
        pickle.dump(record, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','coal','exp2','0.pkl'),'wb'))
        
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

            pickle.dump(record, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','coal',f'{offset+ii_w}.pkl'),'wb'))
            
    if do_=='run_batches':
        
        N_BATCHES = 8
        xs = []
        ys = []
        
        # load the pickled data first
        run_files = glob.glob(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','coal','*.pkl'))

        w_prod = 0.75
        w_trad = 0.25

        for run_file in run_files:

            res = pickle.load(open(run_file, 'rb'))
            
            chroma = params2chroma(res["SCALE_FACTOR"],res['ALPHA'])

            xs.append(chroma)
            ys.append(w_prod*res["weighted_prod_loss"] + w_trad*res["weighted_trade_loss"])
            
        offset = max([int(os.path.split(os.path.splitext(f)[0])[1]) for f in run_files if not 'ini' in f])
        print ('offset:',offset)
        
        N_WORKERS = 8
        STEP_INI = 15
        GAMMA = 0.8
        pool = mp.Pool(N_WORKERS)
        
        for batch in range(N_BATCHES):
                
            xp = np.array(xs)
            yp = np.array(ys)
            
            # Create the GP
            #if gp_params is not None:
            #    model = gp.GaussianProcessRegressor(**gp_params)
            #else:
            kernel = gp.kernels.Matern()
            model = gp.GaussianProcessRegressor(kernel=kernel,
                                                alpha=1e-5,
                                                n_restarts_optimizer=10,
                                                normalize_y=True)
            model.fit(xp, yp)
            
            
            n_params = len(xs[0])
            bounds = np.array([[0,1]]*n_params)
            
            df_alpha_ini = df_alpha.copy()
            
            # generate a bunch of random samples to assess against the gp
            x_random = np.random.uniform(bounds[:,0],bounds[:,1], size=(10000,n_params))
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=False, n_params=n_params)
            print ('ei shape',ei.shape, ei.min(), ei.max())
            
            next_samples = x_random[ei.argsort()[-N_WORKERS:][::-1],:]
            
            print (next_samples)
            
            print ('df_alpha_ini')
            print (df_alpha_ini.values)
            
            #print (next_samples.shape)
            
            differential_fill_params = []
            for ii_w, sample in enumerate(next_samples.tolist()):
                print (len(sample))
                
                df_alpha.loc[df_alpha.index] = 10**(np.array(sample[1:])*3-3)
                #df_alpha = df_alpha_ini + (np.array(sample[1:])-0.5)/10
                print (f'{ii_w} df_alpha')
                print (df_alpha.values)
                
                
                
                SCALE_FACTOR = 10**(2+sample[0]*4)
                differential_fill_params.append(
                    [
                        community_nodes,
                        dijkstra_mincost_adj,  
                        df_alpha,
                        STEP_INI,
                        GAMMA,
                        SCALE_FACTOR, # try some random scaling factors between 10**3 and 10**6
                        params,
                        logging.getLogger(f'dijkstra_fill_worker_{ii_w}'),
                        ii_w,
                    ]
                )
                

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
                print ('results:',ii_w,record['SCALE_FACTOR'], weighted_prod_loss, weighted_trade_loss)

                pickle.dump(record, open(os.path.join(os.getcwd(),'results','interdiction','dijkstra_flow','coal',f'{offset+ii_w}.pkl'),'wb'))
                
                
                
                xs.append(params2chroma(record['SCALE_FACTOR'],record['ALPHA']))
                ys.append(w_prod*record["weighted_prod_loss"] + w_trad*record["weighted_trade_loss"])
                
            offset += N_WORKERS            

    return []
"""
    
def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7):
    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.
    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    n_params = bounds.shape[0]

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    for n in range(n_iters):

        model.fit(xp, yp)

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    return xp, yp
    
