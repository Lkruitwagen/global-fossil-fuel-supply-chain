import os, json
import geopandas as gpd
import pandas as pd
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import pyproj
import pygeos

from shapely import geometry, ops, wkt
from functools import partial

from ffsc.pipeline.nodes.utils import V_inv
    
def chunk_worker(left_spec, right_spec, include_distance, Q_slice):
    
    # recover the SharedMemory bloc
    left_shm = SharedMemory(left_spec['name'])
    right_shm = SharedMemory(right_spec['name'])
    # Create the np.recarray from the buffer of the shared memory
    left_arr = np.recarray(shape=left_spec['shape'], dtype=left_spec['dtype'], buf=left_shm.buf)
    right_arr = np.recarray(shape=right_spec['shape'], dtype=right_spec['dtype'], buf=right_shm.buf)
    
    #print (left_arr[0][2], right_arr[0][2])
    
    
        
    
    results = []
    for idx_pair in Q_slice:
        geom1 = wkt.loads(left_arr[idx_pair[1]][2])
        geom2 = wkt.loads(right_arr[idx_pair[0]][3]) # buffered_geom

        
        if (geom1!=geom2)  and (geom1.intersects(geom2)):
            if include_distance:
                geom3 = wkt.loads(right_arr[idx_pair[0]][2]) # non-buffered geom
                
                pt1,pt2 = ops.nearest_points(geom1, geom3)
                
                dist, a1, a2 = V_inv((pt1.y, pt1.x),(pt2.y, pt2.x))
                
                dist = dist *1000 #m
                
                results.append((idx_pair[0], idx_pair[1], True, pt1.wkt, dist))
                
            else:
                results.append((idx_pair[0], idx_pair[1], True, None, None))
        else:
            results.append((idx_pair[0], idx_pair[1],False,None, None))
        
    return results
    
def to_shm(df, name):
    array = df.reset_index().to_records(index=False)
    shape, dtype = array.shape, array.dtype
    
    shm = SharedMemory(name=name, create=True, size=array.nbytes)
    shm_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
    shm_spec = {'name':name, 'shape':shape, 'dtype':dtype}
    
    np.copyto(shm_array, array)
    
    return shm, shm_spec

def mp_sjoin_mindist(df_linear, df_buffered, left_geom_column, right_geom_column, right_buffer_column, N_workers, logger, include_min_dist=False):
    
    ### get and query tree using pygeos
    logger.info('Getting STRTree and querying')
    tree = pygeos.STRtree([pygeos.io.from_wkt(el) for el in df_linear[left_geom_column].values])
    Q = tree.query_bulk([pygeos.io.from_wkt(el) for el in df_buffered[right_buffer_column].values])
    logger.info(f'Got tree of with {Q.shape[1]} results, now verifying intersections.')
    
    # temp for prototyping
    # Q = Q[:,0:1200]
    
    logger.info('Assigning SharedMemory')
    left_shm, left_spec = to_shm(df_linear[['unique_id',left_geom_column]],'linear')
    right_shm, right_spec = to_shm(df_buffered[['unique_id',right_geom_column, right_buffer_column]],'buffered')
    
    # do multiprocess
    logger.info('Async SJoin')
    chunk = Q.shape[1]//N_workers +1
    
    args = [(left_spec, right_spec, include_min_dist, Q.T[ii*chunk:(ii+1)*chunk,:]) for ii in range(N_workers)]
    
    with mp.Pool(N_workers) as pool:
        res = pool.starmap(chunk_worker, args)
        
    res = [item for sublist in res for item in sublist]
    
    logger.info(f'obtained results {len(res)}')
    
    # release shared mem
    left_shm.close()
    left_shm.unlink()
    right_shm.close()
    right_shm.unlink()
    
    return res
    


def null_forward(df):
    return df