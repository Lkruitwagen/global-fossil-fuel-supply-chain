import logging, os, sys, pickle, json
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
import pandas as pd
from math import pi
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def prep_coal_nx(df_edges_coal, df_cities, df_powerstations, df_coalmines, df_energy_balance, flow_parameters):
    logger=logging.getLogger('prep_coal_nx')
       
    logger.info('Making graph from edges')
    G = nx.DiGraph()
    
    df_edges_coal['IMPEDANCE'] = (df_edges_coal['IMPEDANCE']*100).astype(int)
    
    G.add_edges_from([(r[0],r[1],{'z':r[2]}) for r in df_edges_coal[['START','END','IMPEDANCE']].values.tolist()])
        
    ### add supersink? maybe not.
    
    logger.info('Getting node demand')
    ## add demand
    df_cities['DEMAND'] = df_cities.apply(lambda el: json.loads(el['features'])['total_coal_consumption']/1e3, axis=1) #TJ/yr
    df_powerstations['DEMAND'] = df_powerstations.apply(lambda el: json.loads(el['features'])['capacity_mw']*8760*.6*3.6/1e3, axis=1) #TJ/yr
    df_powerstations['DEMAND'] = df_powerstations['DEMAND'] * df_powerstations.apply(lambda el: 1 if json.loads(el['features'])['fuel1']=='Coal' else 0, axis=1)
    
    min_demand = 10 # min(df_powerstations.loc[df_powerstations['DEMAND']>0,'DEMAND'].min(),df_cities['DEMAND'].min())
    
    nx.set_node_attributes(G, 0, 'D')
    
    city_nodes = [n for n in G.nodes if 'CITY' in n]
    
    logger.info(f'starting cities: {len(df_cities)}')
    
    scope_cities = [rec for rec in df_cities[['unique_id','DEMAND']].to_dict(orient='records') if rec['unique_id'] in city_nodes]
    
    logger.info(f'scope cities{len(scope_cities)}')
    
    attrs = {rec['unique_id']:{'D':int(round(rec['DEMAND']/min_demand))} for rec in scope_cities} # TJ/yr
    attrs = {kk:vv for kk,vv in attrs.items() if vv['D']>0}
    
    nx.set_node_attributes(G, attrs)
    
    logger.info(f'starting powerstations: {len(df_powerstations)}')
       
    pwrstn_nodes = [n for n in G.nodes if 'POWERSTATION' in n]
    
    scope_powerstns = [rec for rec in df_powerstations[['unique_id','DEMAND']].to_dict(orient='records') if rec['unique_id'] in pwrstn_nodes]
    
    logger.info(f'scope_powerstations {len(scope_powerstns)}')
    
    attrs = {rec['unique_id']:{'D':int(round(rec['DEMAND']/min_demand))} for rec in scope_powerstns}
    attrs = {kk:vv for kk,vv in attrs.items() if vv['D']>0}

    nx.set_node_attributes(G, attrs)
    
    
    
    ### add supersource (then drop then add again)
    G.add_node('supersource')
    
    source_nodes = [n for n in G.nodes if 'COALMINE' in n]
    for source_node in source_nodes:
        G.add_edge('supersource',source_node,z=0)
    
    
    if flow_parameters['check_paths']:
        logger.info(f'checking powerstation paths...')

        p_count = 0
        c_count=0

        pathless_pwrstns = []
        for ii_p, rec in enumerate(scope_powerstns):
            if ii_p %1000==0:
                logger.info(f'ii_p {ii_p}, p_count {p_count}')
            if not nx.has_path(G, 'supersource',rec['unique_id']):
                #logger.info(f'No Path! {pwrstn}')
                p_count +=1
                pathless_pwrstns.append(rec['unique_id'])
                
        logger.info(f'checking city paths...')
        pathless_cities = []
        for ii_c, rec in enumerate(scope_cities):
            if ii_c %1000==0:
                logger.info(f'ii_c {ii_c}, c_count {c_count}')
            if not nx.has_path(G, 'supersource',rec['unique_id']):
                #logger.info(f'No Path! {city}')
                c_count+=1
                pathless_cities.append(rec['unique_id'])


        logger.info(f'pathless powerstations: {len(pathless_pwrstns)}')
        logger.info(f'pathless cities:{len(pathless_cities)}')
        
        G.remove_nodes_from(pathless_pwrstns)
        G.remove_nodes_from(pathless_cities)
        

        
        missing_cities = df_cities[df_cities['unique_id'].isin(pathless_cities)]
        missing_powerstations = df_powerstations[df_powerstations['unique_id'].isin(pathless_pwrstns)]
    else:
        missing_cities = pd.DataFrame()
        missing_powerstations = pd.DataFrame()

    D_cities = sum([G.nodes[u].get('D',0) for u in list(G) if 'CITY' in u])
    D_pwrstns = sum([G.nodes[u].get('D',0) for u in list(G) if 'POWERSTATION' in u])

    logger.info(f'sum D pwnstns: {D_pwrstns}, sum D cities: {D_cities}')
    logger.info(f'combined sum {D_pwrstns+D_cities}')

    demand_df = pd.DataFrame.from_dict({u:G.nodes[u].get('D',0) for u in list(G)}, orient='index',columns=['D'])

    all_demand = demand_df['D'].sum()

    attrs = {'supersource':{'D':-1*all_demand}}
    nx.set_node_attributes(G, attrs)
    
    
    if flow_parameters['constrain_production']:       
        
        ### get demand weighting
                
        # remove supersource again
        G.remove_nodes_from(['supersource'])
        
        ### check components here, get demand in connected components
        components = {ii_c:[list(cc)] for ii_c, cc in enumerate(nx.weakly_connected_components(G))}
        df_components = pd.DataFrame.from_dict(components, orient='index', columns=['unique_id']).explode('unique_id').reset_index().rename(columns={'index':'component'})
        print ('df_components')
        print (df_components)
        
        df_components = pd.merge(df_components, demand_df, how='left',left_on='unique_id',right_index=True)
        
        
        # get coalmines in network
        all_mines = [u for u in list(G) if 'COALMINE' in u]
        df_coalmines = df_coalmines[df_coalmines['unique_id'].isin(all_mines)]
        
        # match their iso2s
        df_coalmines['iso2_map'] = df_coalmines['iso2'].apply(lambda el: el if el in df_energy_balance['ISO_A2'].values.tolist() else 'REM')
        
        # get all the scope iso2s
        all_iso2 = df_coalmines.loc[df_coalmines['unique_id'].isin(all_mines),'iso2_map'].unique().tolist()
        
        # drop non-scope iso2
        df_energy_balance = df_energy_balance[df_energy_balance['ISO_A2'].isin(all_iso2)]
        df_energy_balance = df_energy_balance.set_index('ISO_A2')
        
        # convert an column with the energy carrier
        df_energy_balance['E'] = df_energy_balance['Coal*']/df_energy_balance['Coal*'].sum()*all_demand
        
        # get a series of the number of mines per country
        N_iso2 = df_coalmines.groupby('iso2_map').size()
        
        df_coalmines['C'] = df_coalmines['iso2_map'].apply(lambda el: df_energy_balance.at[el,'E']/N_iso2.at[el])
        
        df_components = pd.merge(df_components, df_coalmines[['unique_id','C']], how='left',on='unique_id')
        #print ('components 1')
        #print (df_components)
        df_scaler = df_components[['component','D','C']].groupby('component').sum().reset_index()
        df_scaler['scaler'] = df_scaler['D']/df_scaler['C']
        #print ('scaler')
        #print (df_scaler.sort_values('scaler'))
        
        df_components = pd.merge(df_components, df_scaler[['component','scaler']], how='left',on='component')
        #print ('components2')
        #print (df_components)
        df_coalmines = pd.merge(df_coalmines, df_components[['unique_id','scaler']], how='left',on='unique_id')
        #print ('coalmines1')
        #print (df_coalmines)
        df_coalmines['C'] = df_coalmines['C']*df_coalmines['scaler']
        
        #print ('coalmines2')
        #print (df_coalmines)
        
        
        # reattach supersource
        G.add_node('supersource')
    
        source_nodes = [n for n in G.nodes if 'COALMINE' in n]
        for source_node in source_nodes:
            G.add_edge('supersource',source_node,z=0)
        
        
        attrs = {('supersource',row['unique_id']):{'C':int(row['C'])+1} for idx, row in df_coalmines.iterrows()}
        
        nx.set_edge_attributes(G,attrs)
        
        attrs = {'supersource':{'D':-1*int(all_demand)}}
        nx.set_node_attributes(G, attrs)

    
    all_impedances = [e[2]['z'] for e in G.edges(data=True)]

    print (nx.info(G))
    print ('impedance: mean',np.mean(all_impedances),'max',np.max(all_impedances),'min',np.min(all_impedances))

    logger.info('Solving Flow!')
    
    print ('pandas')
    #print (nx.to_pandas_edgelist(G))
    
    df_edges = nx.to_pandas_edgelist(G)
    df_nodes = pd.DataFrame([{**D,**{'NODE':n}} for n,D in G.nodes(data=True)])
    
    df_edges['C'] = np.ceil(df_edges['C'].fillna(all_demand)).astype(int)
    
    print ('pandas edges')
    print (df_edges)
    
    print ('pandas nodes')
    print (df_nodes)
    
    


    #flow_cost, flow_dict = network_simplex(G, demand='D', capacity='capacity', weight='z')

    #print ('flow cost',flow_cost)
    #print (flow_dict)

    #pickle.dump(flow_cost, open('./flow_cost.pkl','wb'))
    #pickle.dump(flow_dict, open('./flow_dict.pkl','wb'))

    return df_edges, df_nodes, missing_cities, missing_powerstations

def prep_oil_nx(df_edges_oil, df_cities, df_powerstations, df_oilfields, df_oilwells, df_energy_balance, flow_parameters):
    logger=logging.getLogger('prep_oil_nx')
       
    logger.info('Making graph from edges')
    G = nx.DiGraph()
    
    df_edges_oil['IMPEDANCE'] = (df_edges_oil['IMPEDANCE']*100).astype(int)
    
    G.add_edges_from([(r[0],r[1],{'z':r[2]}) for r in df_edges_oil[['START','END','IMPEDANCE']].values.tolist()])
    
    ### add supersource
    source_nodes = [n for n in G.nodes if (('OILFIELD' in n) or ('OILWELL' in n))]
    G.add_node('supersource')

    for source_node in source_nodes:
        G.add_edge('supersource',source_node,z=0)
        
    ### add supersink? maybe not.
    
    logger.info('Getting node demand')
    ## add demand
    df_cities['DEMAND'] = df_cities.apply(lambda el: json.loads(el['features'])['total_oil_consumption']/1e3, axis=1) #TJ/yr
    df_powerstations['DEMAND'] = df_powerstations.apply(lambda el: json.loads(el['features'])['capacity_mw']*8760*.6*3.6/1e3, axis=1) #TJ/yr
    df_powerstations['DEMAND'] = df_powerstations['DEMAND'] * df_powerstations.apply(lambda el: 1 if json.loads(el['features'])['fuel1']=='Oil' else 0, axis=1)
    
    min_demand = 10 # min(df_powerstations.loc[df_powerstations['DEMAND']>0,'DEMAND'].min(),df_cities['DEMAND'].min())
    
    nx.set_node_attributes(G, 0, 'D')
    
    city_nodes = [n for n in G.nodes if 'CITY' in n]
    
    logger.info(f'starting cities: {len(df_cities)}')  # checkpt
    
    scope_cities = [rec for rec in df_cities[['unique_id','DEMAND']].to_dict(orient='records') if rec['unique_id'] in city_nodes]
    
    logger.info(f'scope cities{len(scope_cities)}')
    
    attrs = {rec['unique_id']:{'D':int(round(rec['DEMAND']/min_demand))} for rec in scope_cities} # TJ/yr
    attrs = {kk:vv for kk,vv in attrs.items() if vv['D']>0}
    
    nx.set_node_attributes(G, attrs)
    
    logger.info(f'starting powerstations: {len(df_powerstations)}')
       
    pwrstn_nodes = [n for n in G.nodes if 'POWERSTATION' in n]
    logger.info(f'ps in graph: {len(pwrstn_nodes)}')
    
    scope_powerstns = [rec for rec in df_powerstations[['unique_id','DEMAND']].to_dict(orient='records') if rec['unique_id'] in pwrstn_nodes]
    
    logger.info(f'scope_powerstations {len(scope_powerstns)}')
    
    attrs = {rec['unique_id']:{'D':int(round(rec['DEMAND']/min_demand))} for rec in scope_powerstns}
    attrs = {kk:vv for kk,vv in attrs.items() if vv['D']>0}

    nx.set_node_attributes(G, attrs)
    
    
    if flow_parameters['check_paths']:
        logger.info(f'checking powerstation paths...')

        p_count = 0
        c_count=0

        pathless_pwrstns = []
        for ii_p, rec in enumerate(scope_powerstns):
            if ii_p %1000==0:
                logger.info(f'ii_p {ii_p}, p_count {p_count}')
            if not nx.has_path(G, 'supersource',rec['unique_id']):
                #logger.info(f'No Path! {pwrstn}')
                p_count +=1
                pathless_pwrstns.append(rec['unique_id'])
                
        logger.info(f'checking city paths...')
        pathless_cities = []
        for ii_c, rec in enumerate(scope_cities):
            if ii_c %1000==0:
                logger.info(f'ii_c {ii_c}, c_count {c_count}')
            if not nx.has_path(G, 'supersource',rec['unique_id']):
                #logger.info(f'No Path! {city}')
                c_count+=1
                pathless_cities.append(rec['unique_id'])


        logger.info(f'pathless powerstations: {len(pathless_pwrstns)}')
        logger.info(f'pathless cities:{len(pathless_cities)}')
        
        G.remove_nodes_from(pathless_pwrstns)
        G.remove_nodes_from(pathless_cities)
        
        missing_cities = df_cities[df_cities['unique_id'].isin(pathless_cities)]
        missing_powerstations = df_powerstations[df_powerstations['unique_id'].isin(pathless_pwrstns)]
    else:
        missing_cities = pd.DataFrame()
        missing_powerstations = pd.DataFrame()

    D_cities = sum([G.nodes[u].get('D',0) for u in list(G) if 'CITY' in u])
    D_pwrstns = sum([G.nodes[u].get('D',0) for u in list(G) if 'POWERSTATION' in u])

    logger.info(f'sum D pwnstns: {D_pwrstns}, sum D cities: {D_cities}')
    logger.info(f'combined sum {D_pwrstns+D_cities}')

    all_demand = sum([G.nodes[u].get('D', 0) for u in list(G)])

    attrs = {'supersource':{'D':-1*all_demand}}
    nx.set_node_attributes(G, attrs)
    
    
    if flow_parameters['constrain_production']:       
        
        ### get demand weighting
        
        # get coalmines in network
        all_sources = [u for u in list(G) if (('OILWELL' in u) or ('OILFIELD' in u))]
        df_oilwells = df_oilwells[df_oilwells['unique_id'].isin(all_sources)]
        df_oilfields = df_oilfields[df_oilfields['unique_id'].isin(all_sources)]
        
        # get a single iso2 for oilfields
        df_oilfields['iso2'] = df_oilfields['iso2'].str.split(';').str[0]
        
        # match their iso2s
        df_oilwells['iso2_map'] = df_oilwells['iso2'].apply(lambda el: el if el in df_energy_balance['ISO_A2'].values.tolist() else 'REM')
        df_oilfields['iso2_map'] = df_oilfields['iso2'].apply(lambda el: el if el in df_energy_balance['ISO_A2'].values.tolist() else 'REM')
        
        # get all the scope iso2s
        all_iso2 = df_oilwells.loc[df_oilwells['unique_id'].isin(all_sources),'iso2_map'].unique().tolist() \
                    + df_oilfields.loc[df_oilfields['unique_id'].isin(all_sources),'iso2_map'].unique().tolist() 
        
        # drop non-scope iso2
        df_energy_balance = df_energy_balance[df_energy_balance['ISO_A2'].isin(all_iso2)]
        df_energy_balance = df_energy_balance.set_index('ISO_A2')
        
        # convert an column with the energy carrier
        df_energy_balance['E'] = df_energy_balance['Crude oil']/df_energy_balance['Crude oil'].sum()*all_demand
        
        # get a series of the number of mines per country
        N_iso2 = df_oilwells.groupby('iso2_map').size() + df_oilfields.groupby('iso2_map').size()
        
        df_oilwells['C'] = df_oilwells['iso2_map'].apply(lambda el: df_energy_balance.at[el,'E']/N_iso2.at[el])
        df_oilfields['C'] = df_oilfields['iso2_map'].apply(lambda el: df_energy_balance.at[el,'E']/N_iso2.at[el])
        
        attrs = {('supersource',row['unique_id']):{'C':row['C']} for idx, row in df_oilwells.iterrows()}
        attrs.update({('supersource',row['unique_id']):{'C':row['C']} for idx, row in df_oilfields.iterrows()})
        
        nx.set_edge_attributes(G,attrs)

    
    all_impedances = [e[2]['z'] for e in G.edges(data=True)]

    print (nx.info(G))
    print ('impedance: mean',np.mean(all_impedances),'max',np.max(all_impedances),'min',np.min(all_impedances))

    logger.info('Solving Flow!')
    
    print ('pandas')
    #print (nx.to_pandas_edgelist(G))
    
    df_edges = nx.to_pandas_edgelist(G)
    df_nodes = pd.DataFrame([{**D,**{'NODE':n}} for n,D in G.nodes(data=True)])
    
    df_edges['C'] = np.ceil(df_edges['C'].fillna(all_demand)).astype(int)
    
    print ('pandas edges')
    print (df_edges)
    
    print ('pandas nodes')
    print (df_nodes)
    


    #flow_cost, flow_dict = network_simplex(G, demand='D', capacity='capacity', weight='z')

    #print ('flow cost',flow_cost)
    #print (flow_dict)

    #pickle.dump(flow_cost, open('./flow_cost.pkl','wb'))
    #pickle.dump(flow_dict, open('./flow_dict.pkl','wb'))

    return df_edges, df_nodes, missing_cities, missing_powerstations

def prep_gas_nx(df_edges_gas, df_cities, df_powerstations, df_oilfields, df_oilwells, df_energy_balance, flow_parameters):
    logger=logging.getLogger('prep_gas_nx')
       
    logger.info('Making graph from edges')
    G = nx.DiGraph()
    
    df_edges_gas['IMPEDANCE'] = (df_edges_gas['IMPEDANCE']*100).astype(int)
    
    G.add_edges_from([(r[0],r[1],{'z':r[2]}) for r in df_edges_gas[['START','END','IMPEDANCE']].values.tolist()])
    
    ### add supersource
    source_nodes = [n for n in G.nodes if (('OILFIELD' in n) or ('OILWELL' in n))]
    G.add_node('supersource')

    for source_node in source_nodes:
        G.add_edge('supersource',source_node,z=0)
        
    ### add supersink? maybe not.
    
    logger.info('Getting node demand')
    ## add demand
    df_cities['DEMAND'] = df_cities.apply(lambda el: json.loads(el['features'])['total_gas_consumption']/1e3, axis=1) #TJ/yr
    df_powerstations['DEMAND'] = df_powerstations.apply(lambda el: json.loads(el['features'])['capacity_mw']*8760*.6*3.6/1e3, axis=1) #TJ/yr
    df_powerstations['DEMAND'] = df_powerstations['DEMAND'] * df_powerstations.apply(lambda el: 1 if json.loads(el['features'])['fuel1']=='Gas' else 0, axis=1)
    
    min_demand = 10 # min(df_powerstations.loc[df_powerstations['DEMAND']>0,'DEMAND'].min(),df_cities['DEMAND'].min())
    
    nx.set_node_attributes(G, 0, 'D')
    
    city_nodes = [n for n in G.nodes if 'CITY' in n]
    
    logger.info(f'starting cities: {len(df_cities)}')
    
    scope_cities = [rec for rec in df_cities[['unique_id','DEMAND']].to_dict(orient='records') if rec['unique_id'] in city_nodes]
    
    logger.info(f'scope cities{len(scope_cities)}')
    
    attrs = {rec['unique_id']:{'D':int(round(rec['DEMAND']/min_demand))} for rec in scope_cities} # TJ/yr
    attrs = {kk:vv for kk,vv in attrs.items() if vv['D']>0}
    
    nx.set_node_attributes(G, attrs)
    
    logger.info(f'starting powerstations: {len(df_powerstations)}')
       
    pwrstn_nodes = [n for n in G.nodes if 'POWERSTATION' in n]
    
    scope_powerstns = [rec for rec in df_powerstations[['unique_id','DEMAND']].to_dict(orient='records') if rec['unique_id'] in pwrstn_nodes]
    
    logger.info(f'scope_powerstations {len(scope_powerstns)}')
    
    attrs = {rec['unique_id']:{'D':int(round(rec['DEMAND']/min_demand))} for rec in scope_powerstns}
    attrs = {kk:vv for kk,vv in attrs.items() if vv['D']>0}

    nx.set_node_attributes(G, attrs)
    
    
    if flow_parameters['check_paths']:
        logger.info(f'checking powerstation paths...')

        p_count = 0
        c_count=0

        pathless_pwrstns = []
        for ii_p, rec in enumerate(scope_powerstns):
            if ii_p %1000==0:
                logger.info(f'ii_p {ii_p}, p_count {p_count}')
            if not nx.has_path(G, 'supersource',rec['unique_id']):
                #logger.info(f'No Path! {pwrstn}')
                p_count +=1
                pathless_pwrstns.append(rec['unique_id'])
                
        logger.info(f'checking city paths...')
        pathless_cities = []
        for ii_c, rec in enumerate(scope_cities):
            if ii_c %1000==0:
                logger.info(f'ii_c {ii_c}, c_count {c_count}')
            if not nx.has_path(G, 'supersource',rec['unique_id']):
                #logger.info(f'No Path! {city}')
                c_count+=1
                pathless_cities.append(rec['unique_id'])


        logger.info(f'pathless powerstations: {len(pathless_pwrstns)}')
        logger.info(f'pathless cities:{len(pathless_cities)}')
        
        G.remove_nodes_from(pathless_pwrstns)
        G.remove_nodes_from(pathless_cities)
        
        missing_cities = df_cities[df_cities['unique_id'].isin(pathless_cities)]
        missing_powerstations = df_powerstations[df_powerstations['unique_id'].isin(pathless_pwrstns)]
    else:
        missing_cities = pd.DataFrame()
        missing_powerstations = pd.DataFrame()

    D_cities = sum([G.nodes[u].get('D',0) for u in list(G) if 'CITY' in u])
    D_pwrstns = sum([G.nodes[u].get('D',0) for u in list(G) if 'POWERSTATION' in u])

    logger.info(f'sum D pwnstns: {D_pwrstns}, sum D cities: {D_cities}')
    logger.info(f'combined sum {D_pwrstns+D_cities}')

    all_demand = sum([G.nodes[u].get('D', 0) for u in list(G)])

    attrs = {'supersource':{'D':-1*all_demand}}
    nx.set_node_attributes(G, attrs)
    
    
    if flow_parameters['constrain_production']:       
        
        ### get demand weighting
        
        # get coalmines in network
        all_sources = [u for u in list(G) if (('OILWELL' in u) or ('OILFIELD' in u))]
        df_oilwells = df_oilwells[df_oilwells['unique_id'].isin(all_sources)]
        df_oilfields = df_oilfields[df_oilfields['unique_id'].isin(all_sources)]
        
        # get a single iso2 for oilfields
        df_oilfields['iso2'] = df_oilfields['iso2'].str.split(';').str[0]
        
        # match their iso2s
        df_oilwells['iso2_map'] = df_oilwells['iso2'].apply(lambda el: el if el in df_energy_balance['ISO_A2'].values.tolist() else 'REM')
        df_oilfields['iso2_map'] = df_oilfields['iso2'].apply(lambda el: el if el in df_energy_balance['ISO_A2'].values.tolist() else 'REM')
        
        # get all the scope iso2s
        all_iso2 = df_oilwells.loc[df_oilwells['unique_id'].isin(all_sources),'iso2_map'].unique().tolist() \
                    + df_oilfields.loc[df_oilfields['unique_id'].isin(all_sources),'iso2_map'].unique().tolist() 
        
        # drop non-scope iso2
        df_energy_balance = df_energy_balance[df_energy_balance['ISO_A2'].isin(all_iso2)]
        df_energy_balance = df_energy_balance.set_index('ISO_A2')
        
        # convert an column with the energy carrier
        df_energy_balance['E'] = df_energy_balance['Natural gas']/df_energy_balance['Natural gas'].sum()*all_demand
        
        # get a series of the number of mines per country
        N_iso2 = df_oilwells.groupby('iso2_map').size() + df_oilfields.groupby('iso2_map').size()
        
        df_oilwells['C'] = df_oilwells['iso2_map'].apply(lambda el: df_energy_balance.at[el,'E']/N_iso2.at[el])
        df_oilfields['C'] = df_oilfields['iso2_map'].apply(lambda el: df_energy_balance.at[el,'E']/N_iso2.at[el])
        
        attrs = {('supersource',row['unique_id']):{'C':row['C']} for idx, row in df_oilwells.iterrows()}
        attrs.update({('supersource',row['unique_id']):{'C':row['C']} for idx, row in df_oilfields.iterrows()})
        
        nx.set_edge_attributes(G,attrs)

    
    all_impedances = [e[2]['z'] for e in G.edges(data=True)]

    print (nx.info(G))
    print ('impedance: mean',np.mean(all_impedances),'max',np.max(all_impedances),'min',np.min(all_impedances))

    logger.info('Solving Flow!')
    
    print ('pandas')
    #print (nx.to_pandas_edgelist(G))
    
    df_edges = nx.to_pandas_edgelist(G)
    df_nodes = pd.DataFrame([{**D,**{'NODE':n}} for n,D in G.nodes(data=True)])
    
    df_edges['C'] = np.ceil(df_edges['C'].fillna(all_demand)).astype(int)
    
    print ('pandas edges')
    print (df_edges)
    
    print ('pandas nodes')
    print (df_nodes)
    


    #flow_cost, flow_dict = network_simplex(G, demand='D', capacity='capacity', weight='z')

    #print ('flow cost',flow_cost)
    #print (flow_dict)

    #pickle.dump(flow_cost, open('./flow_cost.pkl','wb'))
    #pickle.dump(flow_dict, open('./flow_dict.pkl','wb'))

    return df_edges, df_nodes, missing_cities, missing_powerstations