import os, sys

import geopandas as gpd
gpd.options.use_pygeos=False
import pandas as pd
import scipy.spatial
import json

from shapely.strtree import STRtree
from shapely import geometry

def cities_delauney():
    """
    do cities<->cities delauney triangles to simulate road network; roads can't cross coastline
    """
    
    paths = {
            'cities-N':          os.path.join(os.getcwd(),'results_backup','simplify','cities_nodes_dataframe.csv'),
            'pipelines-cities':  os.path.join(os.getcwd(),'results_backup','simplify','cities_pipelines_edge_dataframe.csv'),
            'ports-cities':      os.path.join(os.getcwd(),'results_backup','output','cities_ports_edge_dataframe.csv'),
            'railways-cities':   os.path.join(os.getcwd(),'results_backup','simplify','cities_railways_edge_dataframe_alt.csv')}
    
    cities = pd.read_csv(paths['cities-N'])
    ne = gpd.read_file(os.path.join(root,'data','ne','ne_10m_coastline.geojson'))
    ne = ne[~ne.geometry.isna()]
    mp = ne.unary_union
    
    cities['coordinates'] = cities['coordinates'].apply(json.loads)
    
    cities['geometry'] = cities['coordinates'].apply(geometry.Point)
    
    data = dict(zip(range(len(cities)),cities[['CityNodeId:ID(CityNode)','geometry']].values.tolist()))
    
    delTri = scipy.spatial.Delaunay(cities['coordinates'].values.tolist())
    
    edges = set()
    # for each Delaunay triangle
    for n in range(delTri.nsimplex):
        # for each edge of the triangle
        # sort the vertices
        # (sorting avoids duplicated edges being added to the set)
        # and add to the edges set
        edge = sorted([delTri.vertices[n,0], delTri.vertices[n,1]])
        edges.add((edge[0], edge[1]))
        edge = sorted([delTri.vertices[n,0], delTri.vertices[n,2]])
        edges.add((edge[0], edge[1]))
        edge = sorted([delTri.vertices[n,1], delTri.vertices[n,2]])
        edges.add((edge[0], edge[1]))
        
    lss = [{'CityNode:START_ID(CityNode)':data[e[0]][0],'CityNode:END_ID(CityNode)':data[e[1]][0],'geometry':geometry.LineString([data[e[0]][1],data[e[1]][1]])} for e in list(edges)]
    
    tree = STRtree(list(mp))
    
    df = pd.DataFrame.from_records(lss)
    
    root = os.path.abspath(os.path.join(os.getcwd()))
    
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'], crs='epsg:4326')
    
    keep_rows = []
    for ii_r,row in enumerate(gdf.iterrows()):
        intersected = [pp for pp in tree.query(row[1]['geometry']) if pp.intersects(row[1]['geometry'])]
        #print (ii_r)
        if len(intersected)>0:
            keep_rows.append(row[0])
            
    gdf = gdf.loc[gdf.index.isin(keep_rows),:]
    
    pd.DataFrame(gdf).to_csv(os.path.join(root,'results_backup','lastmile','cities_lastmile.csv'))

    