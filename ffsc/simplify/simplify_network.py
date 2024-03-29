import logging, os, sys, pickle, time, glob, json

import networkx as nx
import pandas as pd
import multiprocessing as mp

NWORKERS = mp.cpu_count()//4

logger=logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

tic=time.time()

def gen_keep_nodes(network, params):
    df_names = {
    'pipeline':['port_pipeline_edge_dataframe.csv',
                'cities_pipelines_edge_dataframe.csv',
                'power_station_pipeline_edge_dataframe.csv',
                'lng_pipeline_edge_dataframe.csv',
                'processing_plant_pipeline_edge_dataframe.csv',
                'refinery_pipeline_edge_dataframe.csv',
                'well_pad_pipeline_edge_dataframe.csv',
                'oil_field_edge_dataframe.csv'],
    'railway':['cities_railways_edge_dataframe_alt.csv',
                'coal_mine_railway_edge_dataframe.csv',
                'power_station_railway_edge_dataframe.csv',
                'port_railway_edge_dataframe.csv']
    }

    node_columns= {}

    keep_nodes = []

    for df_name in df_names[network]:

        df = pd.read_csv(os.path.join(os.getcwd(),'results_backup','simplify',df_name))

        node_column = [cc for cc in df.columns if params['colstr'] in cc][0]

        keep_nodes += df[node_column].unique().tolist()
        logger.info(f'getting nodes {df_name}, total: {len(keep_nodes)}')

    pickle.dump(keep_nodes, open(os.path.join(os.getcwd(),'results_backup',network+'_keepnodes.pkl'),'wb'))

    return keep_nodes



def gen_subgraph_nodes(edges_df, keep_nodes, network):

    logger = logging.getLogger(__name__)
    tic = time.time()

    # instantiate the undirected graph object
    G = nx.Graph()

    # fill the graph from the df
    logger.info(f'Filling graph object... {time.time()-tic:.2f}')
    G.add_edges_from([(r[1],r[2],{'distance':r[4]})for r in edges_df.to_records()])

    # logger.info(f'Dumping graph object.. {time.time()-tic:.2f}') ... lol no way, OoM error
    # pickle.dump(G, open(os.path.join(os.getcwd(),'results_backup','edge_graph.pkl'),'wb'))

    # get edges where degree==2
    logger.info(f'Getting degree... {time.time()-tic:.2f}')
    degrees = {node:val for (node, val) in G.degree() if val==2}

    # remove nodes to keep
    simplify_nodes = set(degrees.keys()) - set(keep_nodes)

    # get subgraph node lists
    logger.info(f'getting sugraph nodes... {time.time()-tic:.2f}')
    subgraph_nodes = [g for g in nx.connected_components(G.subgraph(list(simplify_nodes)))]

    # dump the graph object
    logger.info(f'dumping graph nodes... {time.time()-tic:.2f}')
    pickle.dump(subgraph_nodes, open(os.path.join(os.getcwd(),'results_backup',network+'_subgraph_nodes.pkl'),'wb'))

    return subgraph_nodes

def _worker_simplify_edges(ii_mp, edges_df_path, subgraph_nodes, params, outpath):
    """
    This function breaks down the pipeline objects into single LineString objects, then for each region calls a function
    that finds the intersection of pipelines and snap the intersecting points into the pipeline LineString objects.
    :edges_df: network data, pipeline or railways
    :keep_nodes: nodes that are connected to other edge types (i.e. don't drop)
    :return: simplified edges_df
    """
    logger = logging.getLogger(__name__)
    tic = time.time()


    # load the df
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

        if len(subgraph.nodes)>2:

            # get endpoints of component
            end_pts = {node:val for (node, val) in subgraph.degree() if val==1}
            

            if len(list(end_pts.keys()))==2:

                # get nodes to drop
                drop_nodes_subgraph = list(g-set(list(end_pts.keys())))

                # create new edge
                new_dist = sum([e[2]['distance'] for e in subgraph.edges(data=True)])
                #print ('new edge',list(end_pts.keys())[0],list(end_pts.keys())[1])
                new_edge = {
                    params['node_start_col']:list(end_pts.keys())[0],
                    params['node_end_col']:list(end_pts.keys())[1],
                    ':TYPE':params['TYPE'],
                    'distance':new_dist,
                    'impedance':new_dist**2,
                }

                # add to list collection
                drop_nodes += drop_nodes_subgraph
                new_edges += [new_edge]
            else:
                print ('ERROR!', len(end_pts))
                print ('end_pts',end_pts)
                print (subgraph.edges())
                print (subgraph.degree())

    logger.info(f'dumping to pickle... {time.time()-tic:.2f}')
    pickle.dump(drop_nodes, open(os.path.join(outpath,str(ii_mp)+'_drop_nodes.pkl'),'wb'))
    pickle.dump(new_edges, open(os.path.join(outpath,str(ii_mp)+'_new_edges.pkl'),'wb'))

def mp_simplify_graph(subgraph_nodes, pkl_path):

    chunk_len = len(subgraph_nodes)//NWORKERS +1

    # rearrange subgraph_nodes
    ll_subgraph_nodes = [subgraph_nodes[ii_l*chunk_len:(ii_l+1)*chunk_len] for ii_l in range(NWORKERS)]


    # call the starmap
    logger.info(f'running pool... {time.time()-tic:.2f}')

    pool = mp.Pool(NWORKERS)
    results = pool.starmap(_worker_simplify_edges, list(zip(range(NWORKERS),
                                                        [os.path.join(os.getcwd(),'results_backup','output',params['fname_edges'])]*NWORKERS,
                                                        ll_subgraph_nodes,
                                                        [params]*NWORKERS,
                                                        [pkl_path]*NWORKERS )))

    print (results)

def tidy_nodes_edges_dfs(edges_df,nodes_df,params, outpath):

    logger.info(f'loading drop nodes and new edges ...')
    drop_node_files = glob.glob(os.path.join(params['pkl_path'],'*_drop_nodes.pkl'))
    new_edge_files = glob.glob(os.path.join(params['pkl_path'],'*_new_edges.pkl'))

    new_edges = []
    for f in new_edge_files:
        new_edges += pickle.load(open(f,'rb'))
    drop_nodes = []
    for f in drop_node_files:
        drop_nodes += pickle.load(open(f,'rb'))


    new_edges = pd.DataFrame(new_edges)
    new_edges[params['node_start_col']] = new_edges[params['node_start_col']]
    new_edges[params['node_end_col']] = new_edges[params['node_end_col']]

    print ('len new edges',len(new_edges))
    print ('sum r/p',(new_edges[params['node_start_col']]==params['fname_nodes'][0]).sum())

    
    logger.info(f'collecting drop edges...')
    drop_edge_index = edges_df[params['node_start_col']].isin(drop_nodes).values \
    + edges_df[params['node_end_col']].isin(drop_nodes).values
    
    drop_nodes = [node.strip() for node in drop_nodes]
    logger.info(f'adding {len(new_edges)} new edges, dropping {len(drop_nodes)} nodes ...')
    #print (dfs['pipeline_edge_dataframe'][edge_index])
    
    edges_df.drop(index=edges_df[drop_edge_index].index, inplace=True)
    edges_df = edges_df.append(new_edges, ignore_index=True)#

    nodes_df.drop(index=drop_nodes, inplace=True)

    logger.info(f'Writing to disk ...')
    edges_df.to_csv(os.path.join(outpath, params['fname_edges']))
    nodes_df.to_csv(os.path.join(outpath, params['fname_nodes']))



if __name__=="__main__":

    RUN = 'railway'

    if RUN=='pipeline':
        params = {
            'node_start_col':'StartNodeId:START_ID(PipelineNode)',
            'node_end_col':'EndNodeId:END_ID(PipelineNode)',
            'node_index_col':'PipeNodeID:ID(PipelineNode)',
            'TYPE':'PIPELINE_CONNECTION',
            'fname_edges':'pipeline_edge_dataframe.csv',
            'fname_nodes':'pipeline_node_dataframe.csv',
            'pkl_path':os.path.join(os.getcwd(),'results_backup','pipeline_simplify'),
            'colstr':'PipelineNode',
        }
    elif RUN=='railway':
        params = {
            'node_start_col':'StartNodeId:START_ID(RailwayNode)',
            'node_end_col':'EndNodeId:END_ID(RailwayNode)',
            'node_index_col':'RailwayNodeID:ID(RailwayNode)',
            'TYPE':'RAILWAY_CONNECTION',
            'fname_edges':'railway_edge_dataframe.csv',
            'fname_nodes':'railway_nodes_dataframe.csv',
            'pkl_path':os.path.join(os.getcwd(),'results_backup','railway_simplify'),
            'colstr':'RailwayNode',
        }

    logger.info(f'getting keep_nodes. {time.time()-tic:.2f} ...')
    keep_nodes = gen_keep_nodes(RUN,params)
    #keep_nodes = pickle.load(open(os.path.join(os.getcwd(),'results_backup',RUN+'_keepnodes.pkl'),'rb'))

    logger.info(f'got keep_nodes. {time.time()-tic:.2f} loading edges df ...')
    edges_df = pd.read_csv(os.path.join(os.getcwd(),'results_backup','output',params['fname_edges']))

    logger.info(f'got keep_nodes. {time.time()-tic:.2f} loading nodes df ...')
    nodes_df = pd.read_csv(os.path.join(os.getcwd(),'results_backup','output',params['fname_nodes']))
    nodes_df.set_index(params['node_index_col'], inplace=True)

    logger.info(f'got edges df. {time.time()-tic:.2f} getting subgraph nodes ...')
    subgraph_nodes = gen_subgraph_nodes(edges_df, keep_nodes, RUN)
    #subgraph_nodes = pickle.load(open(os.path.join(os.getcwd(),'results_backup',RUN+'_subgraph_nodes.pkl'),'rb'))

    logger.info(f'got subgraph nodes. {time.time()-tic:.2f} running mp simplify ...')
    mp_simplify_graph(subgraph_nodes, params['pkl_path'])

    logger.info(f'tidying edge and node df {time.time()-tic:.2f}')
    outpath = os.path.join(os.getcwd(),'results_backup','simplify')
    tidy_nodes_edges_dfs(edges_df,nodes_df,params, outpath)
