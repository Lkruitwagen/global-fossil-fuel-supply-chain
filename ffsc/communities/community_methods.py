import logging, os, sys, pickle, json
import subprocess

import pandas as pd
import numpy as np
from shapely import geometry, wkt, ops

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def format_textfiles(node_df, edge_df):
    
    node_types = node_df['NODE'].str.split('_').str[0].unique()
    
    ## get carrier
    if 'COALMINE' in node_types:
        carrier='COAL'
    elif 'LNGTERMINAL' in node_types:
        carrier='GAS'
    else:
        carrier='OIL'
        
        
    # maybe drop supersource from edges?
    edge_df = edge_df[(edge_df['source']!='supersource') & (edge_df['target']!='supersource')]
        
    logger = logging.getLogger('COMMUNITIES-'+carrier)
    
    node_df = node_df.reset_index()
    
    edge_df['w'] = np.round(edge_df['z'].max()/(edge_df['z']+1)).astype(int)
    
    edge_df = pd.merge(edge_df, node_df.reset_index()[['index','NODE']], how='left',left_on='source',right_on='NODE').rename(columns={'index':'source_idx'}).drop(columns=['NODE'])
    
    edge_df = pd.merge(edge_df, node_df.reset_index()[['index','NODE']], how='left',left_on='target',right_on='NODE').rename(columns={'index':'target_idx'}).drop(columns=['NODE'])
    
    logger.info('Writing DirectedLouvain text file')
    with open(os.path.join(os.getcwd(),'results','communities',carrier+'.txt'),'w') as f:

        f.writelines([f'{el[0]:d} {el[1]:d} {el[2]:d}\n' for el in edge_df[['source_idx','target_idx','w']].values.tolist()])
            
            
    return []

def run_communities(dummy):
    logger = logging.getLogger('RUN-COMMUNITIES')
    
    community_dir = os.path.join(os.getcwd(),'results','communities')
    bin_dir = os.path.join(os.getcwd(),'DirectedLouvain','bin')
    
    run_graphs = [txtfile for txtfile in ['COAL.txt','OIL.txt','GAS.txt'] if os.path.exists(os.path.join(community_dir,txtfile))]
    
    
    ### convert to binary
    processes = {}
    for txtfile in run_graphs:
        logger.info(f'starting {txtfile}')
        
        cmd = [
            bin_dir+'/convert',
            '-i',
            os.path.join(community_dir, txtfile),
            '-o',
            os.path.join(community_dir,os.path.splitext(txtfile)[0]+'.bin'),
            '-w',
            os.path.join(community_dir, os.path.splitext(txtfile)[0]+'.weights'),
        ]
        
        processes[txtfile] = subprocess.Popen(cmd, stdout = subprocess.PIPE)
    logger.info('conversion running.')
    for kk, vv in processes.items():
        vv.wait()
    logger.info('conversion waiting')
        
    #for kk, vv in processes.items():
    #    for line in iter(vv.stdout.readline,b''):
    #        sys.stdout.write(line.decode())
            
            
    ### run community detection with weights
    processes = {}
    for txtfile in run_graphs:
        logger.info(f'detecting communities {os.path.splitext(txtfile)[0]}')
        
        cmd = [
            bin_dir+'/community',
            os.path.join(community_dir, os.path.splitext(txtfile)[0]+'.bin'),
            '-w',
            os.path.join(community_dir, os.path.splitext(txtfile)[0]+'.weights'),
            '-l',
            '-1',
            '>',
            os.path.join(community_dir, os.path.splitext(txtfile)[0]+'.tree'),
        ]
        #print (' '.join(cmd))
        
        processes[txtfile] = subprocess.Popen(' '.join(cmd), shell=True, stdout = subprocess.PIPE)
    logger.info('communities running.')
    for kk, vv in processes.items():
        vv.wait()
    logger.info('communities waiting')
        
    #for kk, vv in processes.items():
    #    for line in iter(vv.stdout.readline,b''):
    #        sys.stdout.write(line.decode())
            
    ### run heirarching
    for txtfile in run_graphs:
        logger.info(f'getting hierarchies {os.path.splitext(txtfile)[0]}')
        
        cmd = [
            bin_dir+'/hierarchy',
            os.path.join(community_dir, os.path.splitext(txtfile)[0]+'.tree'),
        ]
        #print (' '.join(cmd))
        
        process = subprocess.Popen(cmd, stdout = subprocess.PIPE)
        process.wait()
        
        for ii_l, line in enumerate(iter(process.stdout.readline,b'')):
            sys.stdout.write(line.decode())

    return []

def post_community_nodes(coal_node_df, oil_node_df, gas_node_df):
    logger = logging.getLogger('POST-COMMUNITIES')
    
    community_dir = os.path.join(os.getcwd(),'results','communities')
    bin_dir = os.path.join(os.getcwd(),'DirectedLouvain','bin')
    
    run_graphs = [txtfile for txtfile in ['COAL.tree','GAS.tree','OIL.tree'] if os.path.exists(os.path.join(community_dir,txtfile))]
    
    tree_depth = {}
    
    ### get the tree depth
    for txtfile in run_graphs:
        logger.info(f'getting N-levels {os.path.splitext(txtfile)[0]}')
        
        cmd = [
            bin_dir+'/hierarchy',
            os.path.join(community_dir, os.path.splitext(txtfile)[0]+'.tree'),
        ]
        #print (' '.join(cmd))
        
        process = subprocess.Popen(cmd, stdout = subprocess.PIPE)
        process.wait()
        
        for ii_l, line in enumerate(iter(process.stdout.readline,b'')):
            if ii_l==0:
                tree_depth[txtfile] = int(line.decode().split(' ')[-1])
            sys.stdout.write(line.decode())
            
    print (tree_depth)
    
    for txtfile in run_graphs:
        logger.info(f'getting levels {os.path.splitext(txtfile)[0]}')
        
        for level in range(tree_depth[txtfile]):
            
            logger.info(f'getting level {level}')
            cmd = [
                bin_dir+'/hierarchy',
                os.path.join(community_dir, os.path.splitext(txtfile)[0]+'.tree'),
                '-l',
                str(level),
                '>',
                os.path.join(community_dir, os.path.splitext(txtfile)[0]+f'_tree_{level}'+'.txt'),
            ]
            
            process = subprocess.Popen(' '.join(cmd), shell=True, stdout = subprocess.PIPE)
            process.wait()
            
    ## then merge
    dfs = {}
    for txtfile in run_graphs:
        logger.info(f'io df levels {os.path.splitext(txtfile)[0]}')
        
        dfs[txtfile] = []
        
        for level in range(tree_depth[txtfile]):
            
            dfs[txtfile].append(
                pd.read_csv(
                    os.path.join(community_dir, os.path.splitext(txtfile)[0]+f'_tree_{level}'+'.txt'), 
                    delim_whitespace=True, 
                    header=None, 
                    index_col=0, 
                    names=['idx',f'comm_{level}'])
            )
            
    logger.info(f'joining df levels {os.path.splitext(txtfile)[0]}')
    if 'COAL.tree' in run_graphs:
        coal_node_df = pd.merge(coal_node_df, pd.concat(dfs['COAL.tree'], axis=1), left_index=True, right_index=True)
    if 'GAS.tree' in run_graphs:
        gas_node_df = pd.merge(gas_node_df, pd.concat(dfs['GAS.tree'], axis=1), left_index=True, right_index=True)
    if 'OIL.tree' in run_graphs:
        oil_node_df = pd.merge(oil_node_df, pd.concat(dfs['OIL.tree'], axis=1), left_index=True, right_index=True)
    
    # return multiple dfs
    return coal_node_df, oil_node_df, gas_node_df

def post_community_edges(params, df_edges, refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations, df_nodes):

    
    print (df_nodes)
    
    df_nodes['NODETYPE'] = df_nodes['NODE'].str.split('_').str[0]
    
    if 'COALMINE' in df_nodes['NODETYPE'].unique():
        carrier='coal'
    elif 'LNGTERMINAL' in df_nodes['NODETYPE'].unique():
        carrier='gas'
    else:
        carrier='oil'
        
    logger=logging.getLogger('Post-community edges '+carrier)
    logger.info('Combining point assets and mapping geometries')
        
    comm_col = f'comm_{params["community_levels"][carrier]-1}'
    
    # rm supersource
    df_edges = df_edges[(df_edges['source']!='supersource') & (df_edges['target']!='supersource')]
    
    # join communities to nodes
    # map geometries onto nodes
    df_ptassets = pd.concat([refineries, oilfields, oilwells, coalmines, lngterminals, ports, cities, powerstations])
    
    df_ptassets['geometry'] = df_ptassets['geometry'].apply(wkt.loads)
    df_ptassets['asset_type']=df_ptassets['unique_id'].str.split('_').str[0]
    df_ptassets.loc[df_ptassets['asset_type'].isin(['CITY','OILFIELD']),'geometry'] = df_ptassets.loc[df_ptassets['asset_type'].isin(['CITY','OILFIELD']),'geometry'].apply(lambda geom: geom.representative_point())
    
    logger.info('merging to nodes')
    df_nodes['NODE_BASE'] = df_nodes['NODE'].str.replace('_B','')
    df_nodes = pd.merge(df_nodes, df_ptassets[['unique_id','geometry']],how='left',left_on='NODE_BASE',right_on='unique_id').drop(columns=['unique_id'])
    print (df_nodes)
    df_nodes.loc[df_nodes['geometry'].isna(),'geometry'] = df_nodes.loc[df_nodes['geometry'].isna(),'NODE'].apply(lambda el: geometry.Point([float(cc) for cc in el.split('_')[2:4]]))
    
    #df_edges['source_id'] = df_edges['source'].str.replace('_B','')
    #df_edges['target_id'] = df_edges['target'].str.replace('_B','')
    
    logger.info('merging to edges')
    df_edges = pd.merge(df_edges, df_nodes[['NODE','geometry',comm_col]], how='left',left_on='source',right_on='NODE').rename(columns={'geometry':'source_geometry',comm_col:'source_comm'}).drop(columns=['NODE'])
    df_edges = pd.merge(df_edges, df_nodes[['NODE','geometry',comm_col]], how='left',left_on='target',right_on='NODE').rename(columns={'geometry':'target_geometry',comm_col:'target_comm'}).drop(columns=['NODE'])
    print ('bork')
    print (df_edges[df_edges[['source_geometry','target_geometry']].isna().any(axis=1)])
    
    df_edges['geometry'] = df_edges.apply(lambda row:geometry.LineString([row['source_geometry'],row['target_geometry']]), axis=1)
    
    logger.info('reducing to communities')
    df_communities = df_nodes.groupby(comm_col).size().rename('N_NODES')
    df_communities = pd.merge(df_communities, df_nodes.groupby(comm_col)['NODETYPE'].apply(lambda ll: list(set(list(ll)))).rename('NODE_TYPES'), how='left',left_index=True, right_index=True)
    df_communities = pd.merge(df_communities, df_nodes.groupby(comm_col)['geometry'].apply(lambda ll: geometry.MultiPoint(list(ll))).rename('geometry'), how='left',left_index=True, right_index=True)
    df_communities['supply'] = df_communities['NODE_TYPES'].str.join(',').str.contains('COALMINE') | df_communities['NODE_TYPES'].str.join(',').str.contains('OILFIELD') | df_communities['NODE_TYPES'].str.join(',').str.contains('OILWELL') 
    df_communities['demand'] = df_communities['NODE_TYPES'].str.join(',').str.contains('CITY') | df_communities['NODE_TYPES'].str.join(',').str.contains('POWERSTATION') 
    
    df_communities = df_communities.sort_values('N_NODES')
    
    print ('df nodes',df_nodes.columns)
    print ('df_edges',df_edges.columns)
    print ('df_communites', df_communities.columns)
    
    logger.info('dump geoms back to str')
    df_edges = df_edges.drop(columns=['source_geometry','target_geometry'])
    df_nodes['geometry'] = df_nodes['geometry'].apply(lambda el: el.wkt)
    df_edges['geometry'] = df_edges['geometry'].apply(lambda el: el.wkt)
    df_communities['geometry'] = df_communities['geometry'].apply(lambda el: el.wkt)
    
    
    return df_nodes, df_edges, df_communities

