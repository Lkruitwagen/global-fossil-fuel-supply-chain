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
    
def chunk_worker(left_spec, right_spec, include_geometry, Q_slice):
    
    # recover the SharedMemory bloc
    left_shm = SharedMemory(left_spec['name'])
    right_shm = SharedMemory(right_spec['name'])
    # Create the np.recarray from the buffer of the shared memory
    left_arr = np.recarray(shape=left_spec['shape'], dtype=left_spec['dtype'], buf=left_shm.buf)
    right_arr = np.recarray(shape=right_spec['shape'], dtype=right_spec['dtype'], buf=right_shm.buf)
    
    #print (left_arr[0][2], right_arr[0][2])
    
    def do_point(geom):
        return [(pygeos.geometry.get_x(geom), pygeos.geometry.get_y(geom))]
    
    def do_multipoint(geom):
        geom_pts = []
        for ii in range(int(pygeos.geometry.get_num_points(geom))):
            pt = pygeos.geometry.get_point(geom, ii)
            geom_pts+=do_point(pt)
        return geom_pts
        
    def do_geomcollection(geom):
        geom_pts = []
        for ii in range(int(pygeos.geometry.get_num_geometries(geom))):
            _geom = pygeos.geometry.get_geometry(geom, ii)
            if pygeos.geometry.get_type_id(_geom)==0:
                geom_pts += do_point(_geom)
            elif pygeos.geometry.get_type_id(_geom)==4:
                geom_pts += do_multipoint(_geom)
        return geom_pts
        
    
    results = []
    for idx_pair in Q_slice:
        geom1 = pygeos.io.from_wkt(left_arr[idx_pair[0]][2])
        geom2 = pygeos.io.from_wkt(right_arr[idx_pair[1]][2])
        
        if (geom1!=geom2)  and (pygeos.predicates.intersects(geom1, geom2)):
            if include_geometry:
                intersection_geom = pygeos.set_operations.intersection(geom1, geom2)
                #results.append((idx_pair[0], idx_pair[1], True, pygeos.io.to_wkt(intersection_geom)))
                
                # Get geometry pts
                if pygeos.geometry.get_type_id(intersection_geom) in [0,4,7]:
                    if pygeos.geometry.get_type_id(intersection_geom)==0: # point
                        geom_pts = do_point(intersection_geom)
                    elif pygeos.geometry.get_type_id(intersection_geom)==4: #MultiPoint
                        geom_pts = do_multipoint(intersection_geom)
                    elif pygeos.geometry.get_type_id(intersection_geom)==7: #geom_collection
                        geom_pts = do_geomcollection(intersection_geom)
                        
                    results.append((idx_pair[0], idx_pair[1], True, geom_pts))
                #else:
                #    pass
                #    #print ('bork!', intersection_geom)
                
                
            else:
                results.append((idx_pair[0], idx_pair[1], True, None))
        else:
            results.append((idx_pair[0], idx_pair[1],False,None))
        
    return results
    
def to_shm(df, name):
    array = df.reset_index().to_records(index=False)
    shape, dtype = array.shape, array.dtype
    
    shm = SharedMemory(name=name, create=True, size=array.nbytes)
    shm_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
    shm_spec = {'name':name, 'shape':shape, 'dtype':dtype}
    
    np.copyto(shm_array, array)
    
    return shm, shm_spec

def mp_sjoin_dissolve(df_left, df_right, left_geom_column, right_geom_column, N_workers, logger, include_geometry=False):
    
    ### get and query tree using pygeos
    logger.info('Getting STRTree and querying')
    tree = pygeos.STRtree([pygeos.io.from_wkt(el) for el in df_right[right_geom_column].values])
    Q = tree.query_bulk([pygeos.io.from_wkt(el) for el in df_left[left_geom_column].values])
    logger.info(f'Got tree of with {Q.shape[1]} results, now verifying intersections.')
    
    # temp for prototyping
    # Q = Q[:,0:1200]
    
    logger.info('Assigning SharedMemory')
    left_shm, left_spec = to_shm(df_left[['unique_id',left_geom_column]],'left')
    right_shm, right_spec = to_shm(df_right[['unique_id',right_geom_column]],'right')
    
    # do multiprocess
    logger.info('Async SJoin')
    chunk = Q.shape[1]//N_workers +1
    
    args = [(left_spec, right_spec, include_geometry, Q.T[ii*chunk:(ii+1)*chunk,:]) for ii in range(N_workers)]
    
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