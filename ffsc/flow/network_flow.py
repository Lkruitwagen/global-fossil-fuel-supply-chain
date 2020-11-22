import logging, os, sys, pickle, json

import networkx as nx
import pandas as pd
from math import pi
import numpy as np

from ffsc.flow.simplex import network_simplex

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def solve_coal(df_edges_coal, df_cities, df_powerstations, df_coalmines, df_energy_balance, flow_parameters):
    logger=logging.getLogger('Solve coalmines')
    
    print (df_powerstations)
    print (df_cities)
    
    logger.info('Making graph from edges')
    G = nx.DiGraph()
    
    G.add_edges_from([(r[0],r[1],{'z':r[2]}) for r in df_edges_coal[['START','END','IMPEDANCE']].values.tolist()])
    
    ### add supersource
    source_nodes = [n for n in G.nodes if 'COALMINE' in n]
    G.add_node('supersource')

    for source_node in source_nodes:
        G.add_edge('supersource',source_node,z=0)
        
    ### add supersink? maybe not.
    
    logger.info('Getting node demand')
    ## add demand
    df_cities['DEMAND'] = df_cities.apply(lambda el: json.loads(el['features'])['total_coal_consumption']/1e3, axis=1) #TJ/yr
    df_powerstations['DEMAND'] = df_powerstations.apply(lambda el: json.loads(el['features'])['capacity_mw']*8760*.6*3.6/1e3, axis=1) #TJ/yr
    df_powerstations['DEMAND'] = df_powerstations['DEMAND'] * df_powerstations.apply(lambda el: 1 if json.loads(el['features'])['fuel1']=='Coal' else 0, axis=1)
    
    print (df_powerstations)
    
    nx.set_node_attributes(G, 0, 'D')
    
    city_nodes = [n for n in G.nodes if 'CITY' in n]
    
    logger.info(f'starting cities: {len(df_cities)}')
    
    scope_cities = [rec for rec in df_cities[['unique_id','DEMAND']].to_dict(orient='records') if rec['unique_id'] in city_nodes]
    
    logger.info(f'scope cities{len(scope_cities)}')
    
    attrs = {rec['unique_id']:{'D':int(round(rec['DEMAND']))} for rec in scope_cities} # TJ/yr
    
    nx.set_node_attributes(G, attrs)
    
    logger.info(f'starting powerstations: {len(df_powerstations)}')
       
    pwrstn_nodes = [n for n in G.nodes if 'POWERSTATION' in n]
    
    scope_powerstns = [rec for rec in df_powerstations[['unique_id','DEMAND']].to_dict(orient='records') if rec['unique_id'] in pwrstn_nodes]
    
    logger.info(f'scope_powerstations {len(scope_powerstns)}')
    

    attrs = {rec['unique_id']:{'D':int(round(rec['DEMAND']))} for rec in scope_powerstns}

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

        print (pathless_pwrstns)
                
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
        
    if flow_parameters['constrain_production']:
        pass


    D_cities = sum([G.nodes[u].get('D',0) for u in list(G) if 'CITY' in u])
    D_pwrstns = sum([G.nodes[u].get('D',0) for u in list(G) if 'POWERSTATION' in u])

    logger.info(f'sum D pwnstns: {D_pwrstns}, sum D cities: {D_cities}')
    logger.info(f'combined sum {D_pwrstns+D_cities}')


    attrs = {'supersource':{'D':-1*(D_pwrstns+D_cities)}}
    nx.set_node_attributes(G, attrs)

    logger.info('Solving Flow!')
    
    flow_cost, flow_dict = network_simplex(G, demand='D', capacity='capacity', weight='z')

    print ('flow cost',flow_cost)
    #print (flow_dict)

    pickle.dump(flow_cost, open('./flow_cost.pkl','wb'))
    pickle.dump(flow_dict, open('./flow_dict.pkl','wb'))

    return None