import glob, os, sys

import geopandas as gpd
import pandas as pd

def combine_euclidean(paths):
    pass

def combine_gdfs(paths):
    gdfs = []
    for f in sorted(paths):
        kk = f.split('/')[-1].split('.')[0]
        print (kk)
        gdfs.append(gpd.read_file(f))
        
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
        
    

if __name__=="__main__":
    
    print ('Doing energy gdfs...')
    energy_paths = glob.glob(os.path.join(os.getcwd(),'data','GHSL_UCDB_ENERGY','*.gpkg'))
    
    gdf = combine_gdfs(energy_paths)
    
    print (gdf)
    gdf.to_file(os.path.join(os.getcwd(),'data','assets','cities_energy.geojson'), driver='GeoJSON')
    
    print ('Doing euclid gdfs...')
    euc_paths = glob.glob(os.path.join(os.getcwd(),'data','GHSL_UCDB_EUCLID','*.gpkg'))
    
    gdf = combine_gdfs(energy_paths)
    
    print (gdf)
    gdf.to_file(os.path.join(os.getcwd(),'data','assets','cities_euclid.geojson'), driver='GeoJSON')

    