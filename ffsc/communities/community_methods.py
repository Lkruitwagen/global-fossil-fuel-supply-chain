import logging, os, sys, pickle, json
import subprocess

import pandas as pd
import numpy as np

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
        
    logger = logging.getLogger('COMMUNITIES-'+carrier)
    
    edge_df['w'] = np.round(edge_df['z'].max()/(edge_df['z']+1)).astype(int)
    
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
    
    run_graphs = [txtfile for txtfile in ['COAL.txt'] if os.path.exists(os.path.join(community_dir,txtfile))]
    
    
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

def post_communities(coal_node_df, oil_node_df, gas_node_df):
    # return multiple dfs
    return coal_node_df, oil_node_df, gas_node_df