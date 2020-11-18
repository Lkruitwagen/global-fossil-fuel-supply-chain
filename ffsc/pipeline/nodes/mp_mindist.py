import os, json
import geopandas as gpd
import pandas as pd
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import pyproj
import pygeos

from shapely import geometry, ops, wkt
from itertools import product

from ffsc.pipeline.nodes.utils import V_inv
    
def chunk_worker(left_spec, right_spec, pairs_slice):
    
    # recover the SharedMemory bloc
    left_shm = SharedMemory(left_spec['name'])
    right_shm = SharedMemory(right_spec['name'])
    # Create the np.recarray from the buffer of the shared memory
    left_arr = np.recarray(shape=left_spec['shape'], dtype=left_spec['dtype'], buf=left_shm.buf)
    right_arr = np.recarray(shape=right_spec['shape'], dtype=right_spec['dtype'], buf=right_shm.buf)
    
    #print (left_arr[0][2], right_arr[0][2])
    
    
        
    
    results = []
    for idx_pair in pairs_slice:
        geom1 = wkt.loads(left_arr[idx_pair[0]][2])
        geom2 = wkt.loads(right_arr[idx_pair[1]][2])
        
        pt1,pt2 = ops.nearest_points(geom1, geom3)
                
        dist, a1, a2 = V_inv((pt1.y, pt1.x),(pt2.y, pt2.x))
                
        dist = dist *1000 #m
                
        results.append((idx_pair[0], idx_pair[1], dist))
                
        
    return results
    
def to_shm(df, name):
    array = df.reset_index().to_records(index=False)
    shape, dtype = array.shape, array.dtype
    
    shm = SharedMemory(name=name, create=True, size=array.nbytes)
    shm_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
    shm_spec = {'name':name, 'shape':shape, 'dtype':dtype}
    
    np.copyto(shm_array, array)
    
    return shm, shm_spec

def mp_get_mindist(df_left, df_right, left_geom_column, right_geom_column, N_workers, logger):
    
    ### get and query tree using pygeos
    
    logger.info('Assigning SharedMemory')
    left_shm, left_spec = to_shm(df_left[['unique_id',left_geom_column]],'left')
    right_shm, right_spec = to_shm(df_right[['unique_id',right_geom_column]],'right')
    
    # do multiprocess
    logger.info('Async mindist')
    chunk = len(df_left)//N_workers +1
    
    all_pairs = list(product(list(range(len(df_left))), list(range(len(df_right)))))
    
    print (len(all_pairs))
    
    args = [(left_spec, right_spec, all_pairs[ii*chunk:(ii+1)*chunk]) for ii in range(N_workers)]
    
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