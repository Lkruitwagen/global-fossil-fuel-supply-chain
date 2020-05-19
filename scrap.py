import logging, os, sys, pickle, time

import networkx as nx
import pandas as pd
import multiprocessing as mp

NWORKERS = mp.cpu_count()//2

logger=logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

tic=time.time()

logger.info(f'Loading the edge df... {time.time()-tic:.2f}')
edges_df = pd.read_csv(edges_df_path)

# instatiate the undirected graph object
G = nx.Graph()

# fill the graph from the df
logger.info(f'Filling graph object... {time.time()-tic:.2f}')
G.add_edges_from([(r[1],r[2],{'distance':r[4]})for r in edges_df.to_records()])

## for each subgraph, simplify
drop_nodes = []
new_edges = []
logger.info(f'n sugraph nodes {len(subgraph_nodes)}... {time.time()-tic:.2f}')

for ii_g, g in enumerate(subgraph_nodes):
    if ii_g % 100==0:
        logger.info(f'ii_g {ii_g}... {time.time()-tic:.2f}')
    # get subgraph
    subgraph = G.subgraph(g)

    # get endpoints of component
    end_pts = {node:val for (node, val) in subgraph.degree() if val==1}

    # get nodes to drop
    drop_nodes_subgraph = list(g-set(end_pts.keys()))

    # create new edge
    new_dist = sum([e[2]['distance'] for e in subgraph.edges(data=True)])
    new_edge = {
        params['node_start_col']:list(end_pts)[0],
        params['node_end_col']:list(end_pts)[1],
        ':TYPE':params['TYPE'],
        'distance':new_dist,
        'impedance':new_dist**2,
    }

    # add to list collection
    drop_nodes += drop_nodes_subgraph
    new_edges += [new_edge]