import os, json
import geopandas as gpd
import pandas as pd
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import pyproj

from shapely import geometry, ops, wkt
from functools import partial

def get_utm_zone(lat,lon):
    """A function to grab the UTM zone number for any lat/lon location
    """
    zone_str = str(int((lon + 180)/6) + 1)

    if ((lat>=56.) & (lat<64.) & (lon >=3.) & (lon <12.)):
        zone_str = '32'
    elif ((lat >= 72.) & (lat <84.)):
        if ((lon >=0.) & (lon<9.)):
            zone_str = '31'
        elif ((lon >=9.) & (lon<21.)):
            zone_str = '33'
        elif ((lon >=21.) & (lon<33.)):
            zone_str = '35'
        elif ((lon >=33.) & (lon<42.)):
            zone_str = '37'

    return zone_str

def buffer_geom(geom, buffer_dist):
    geom_wgs = wkt.loads(geom)
    pt = geom_wgs.representative_point()
    utm_zone = get_utm_zone(pt.y,pt.x)   

    PROJ_WGS = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    PROJ_UTM = pyproj.Proj(proj='utm',zone=utm_zone,ellps='WGS84')
    
    REPROJ_WGS_UTM = partial(pyproj.transform, PROJ_WGS, PROJ_UTM)
    REPROJ_UTM_WGS = partial(pyproj.transform, PROJ_UTM, PROJ_WGS)
    
    utm_geom = ops.transform(REPROJ_WGS_UTM, geom_wgs)
    utm_geom_buffer = utm_geom.buffer(buffer_dist)
    
    wgs_geom_buffer = ops.transform(REPROJ_UTM_WGS, utm_geom_buffer)
    
    if utm_zone in ['1','60']:
        # check IDL
        neg_box = geometry.box(-179.999999999,-90,-179,90)
        pos_box = geometry.box(179,-90,179.999999999,90)
        
        if wgs_geom_buffer.intersects(neg_box) and wgs_geom_buffer.intersects(pos_box):
            # run roh IDL intersection
            if wgs_geom_buffer.type=='MultiPolygon':
                wgs_geom_buffer = list(wgs_geom_buffer)
            elif wgs_geom_buffer.type=='Polygon':
                wgs_geom_buffer=[wgs_geom_buffer]
            new_polys = []
            for pp in wgs_geom_buffer:
                new_polys += [
                    geometry.Polygon([cc for cc in pp.exterior.coords[:] if cc[0]<0]),
                    geometry.Polygon([cc for cc in pp.exterior.coords[:] if cc[0]>0])
                ]
                
            wgs_geom_buffer = geometry.MultiPolygon(new_polys)
    
    return wgs_geom_buffer
    
def buffer_worker(shm_spec, idxs, buffer_dist):
    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    
    # recover the SharedMEmory block
    shm = SharedMemory(shm_spec['name'])

    # Create the np.recarray from the buffer of the shared memory
    async_arr = np.recarray(shape=shm_spec['shape'], dtype=shm_spec['dtype'], buf=shm.buf)
    
    results = []
    for idx in idxs:
        results.append((idx, buffer_geom(async_arr[idx][1], buffer_dist)))
        
    return results
    
def mp_buffer(df,buffer_dist,n_workers):
      
    # array to sharedmem
    array = df[['unique_id','geometry']].to_records(index=False)
    shape, dtype = array.shape, array.dtype
    shm = SharedMemory(name='arr',create=True, size=array.nbytes)
    shm_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
    np.copyto(shm_array, array)
    
    shm_spec = {'name':'arr','shape':shape,'dtype':dtype}
    
    # do multiprocess
    chunk = len(df)//n_workers +1
    
    args = [(shm_spec, range(len(df))[ii*chunk:(ii+1)*chunk], buffer_dist) for ii in range(n_workers)]
    
    with mp.Pool(n_workers) as pool:
        res = pool.starmap(buffer_worker, args)
        
    res = [item for sublist in res for item in sublist]
    
    shm.close()
    shm.unlink()
        
    return [r[1] for r in res]
