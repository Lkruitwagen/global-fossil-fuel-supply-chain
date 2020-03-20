import glob, os, sys, random, logging, warnings

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import geometry
import numpy as np
from geovoronoi import polygon_lines_from_voronoi,polygon_shapes_from_voronoi_lines, coords_to_points, assign_points_to_voronoi_polygons
from scipy.spatial import Voronoi
from shapely.strtree import STRtree
from shapely.ops import cascaded_union, polygonize


warnings.filterwarnings('ignore')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# In this file we load the UCDB and voronoi-buffer it

# Re-save everything down to country, so multi-country UCs just get split to two. Can recombine later if makes sense.

class EuclideanAllocation:

    def __init__(self, countries_path, ucdb_path, ne_path,save_dir):
        logging.info(f'init...')
        self.iso2_df = pd.read_csv(countries_path).set_index('iso2')
        self.ucdb = gpd.read_file(ucdb_path)
        self.ne = gpd.read_file(ne_path)
        #print (self.ucdb['geometry'].type.unique()) -> all multipolygon
        self.dist_thresh=0.014
        self.save_dir = save_dir
        logging.info(f'saving to {self.save_dir}')
        self.wgs_box = geometry.box(-180,-90,180,90)




        # run voronoi
        # grid geom
        # query the edgars CO2 by end-use
        # get coal/oil/gas factors from IEA
        # get coal/oil/gas per UC

    def run_country(self, country_iso2, country_iso3=None):
        if country_iso2:
            self.iso2 = country_iso2
            self.iso3 = self.iso2_df.at[self.iso2,'iso3']
        elif country_iso3:
            self.iso3 = country_iso3
            print (self.iso2_df[self.iso2_df.iso3==self.iso3])
            self.iso2 = self.iso2_df[self.iso2_df.iso3==self.iso3].iloc[0].name
        logging.info(f'Running country: {self.iso2, self.iso3}')

        country_shape = self.ne[self.ne.ISO_A3==self.iso3]

        country_shape = country_shape.iloc[0].geometry
        
        ucdb_slice = self.ucdb[self.ucdb.XC_ISO_LST.str.contains(self.iso3)]

        ucdb_slice['geometry'] = ucdb_slice['geometry'].buffer(0)

        # Clip off country border
        ucdb_slice['geometry'] = ucdb_slice['geometry'].intersection(country_shape)

        ucdb_slice = ucdb_slice[~ucdb_slice['geometry'].is_empty]

        ucdb_slice, voronoi_shapes, all_coords = self.voronoi_ucs(ucdb_slice, country_shape)

        all_ucdb_mp = cascaded_union(ucdb_slice['geometry'])


        ucdb_slice.loc[:,'exp_geom'] = ucdb_slice.apply(lambda row: self._rm_mp(row,all_ucdb_mp), axis=1)


        vs_gdf = gpd.GeoDataFrame(None,geometry=voronoi_shapes)


        fig, ax = plt.subplots(1,1,figsize=(24,24))
        vs_gdf.plot(alpha=0.2,facecolor='none',edgecolor='k', ax=ax)
        #ax.scatter(np.array(all_coords)[:,0],np.array(all_coords)[:,1], color='r')
        ucdb_slice.plot(alpha=0.8, facecolor='none',edgecolor='g', ax=ax)
        ucdb_slice.set_geometry('exp_geom').plot(alpha=0.5, facecolor='none',edgecolor='r',ax=ax)
        bbox = {'alpha':0.5,'lw':None,'color':'w'}

        for row in ucdb_slice.iterrows():
            ax.text(row[1]['exp_geom'].centroid.x, row[1]['exp_geom'].centroid.y, row[1]['UC_NM_LST'], bbox=bbox)

        fig.savefig(os.path.join(self.save_dir,self.iso2+'.png'), dpi=200)
        
        ucdb_slice['geom_gj'] = ucdb_slice.apply(lambda row: row['geometry'].__geo_interface__, axis=1)
        ucdb_slice = ucdb_slice.set_geometry('exp_geom').drop(columns=['geometry'])

        ucdb_slice.to_file(os.path.join(self.save_dir,self.iso2+'.gpkg'),driver='GPKG')


    def _tree_search_union(self,tree,row):
        result = tree.query(row['geometry'])
        result = [r for r in result if r.intersects(row['geometry'])]
        result.append(row['geometry'])
        return cascaded_union(result)

    def _rm_mp(self,row,mp):
        return row['exp_geom'].difference(mp.difference(row['geometry']))

    def _resample_coords(self,coords):
        new_coords = []
        new_coords.append(coords[0])
        for ii_c in range(1,len(coords)):
            dist= np.linalg.norm(np.array(coords[ii_c])-np.array(new_coords[-1]))
            while dist>self.dist_thresh:

                n = np.array(coords[ii_c])-np.array(new_coords[-1])
                n /= np.linalg.norm(n) #unit vector
                new_coord = np.array(new_coords[-1])+n * 0.01
                new_coords.append(new_coord)
                dist= np.linalg.norm(np.array(coords[ii_c])-np.array(new_coord))

            new_coords.append(coords[ii_c])

        return new_coords

    def voronoi_ucs(self, ucdb_slice,country_shape):
        """
        return the slice but with an extra column for the extended area.
        """

        all_coords = []
        for row in ucdb_slice.iterrows():

            if row[1].geometry.type=='Polygon':# and not row[1].geometry.is_empty:
                all_coords += self._resample_coords(row[1].geometry.exterior.coords)

            elif row[1].geometry.type=='MultiPolygon':# and not row[1].geometry.is_empty:
                for subgeom in list(row[1].geometry):
                    all_coords += self._resample_coords(subgeom.exterior.coords)

        vor_coords = np.array(all_coords)

        vor = Voronoi(vor_coords)

        poly_lines = polygon_lines_from_voronoi(vor, country_shape)

        """
        fig, ax = plt.subplots(1,1,figsize=(6,6))

        print ('VLIDDd')
        print (country_shape.is_valid)
        for l in poly_lines:
            xs,ys = l.xy
            ax.plot(xs,ys,c='g')
            #for subpp in list(country_shape):
            #    if subpp.intersects(l):
            #        xs,ys = subpp.exterior.xy
            #        ax.plot(xs,ys,c='r')
        plt.show()
        """

        fix_poly_lines = []

        for l in poly_lines:
            if l.intersects(self.wgs_box.exterior):
                valid_coords = [c for c in l.coords if geometry.Point(c).within(self.wgs_box)]
                if len(valid_coords)>2:
                    fix_poly_lines.append(geometry.LineString(valid_coords))
            else:
                fix_poly_lines.append(l)


        """
        for l in polygonize(poly_lines):
            try:
                print (l.bounds, l.is_valid, l.intersection(country_shape).is_valid)
            except:
                fig, ax = plt.subplots(1,1,figsize=(6,6))
                xs,ys = l.exterior.xy
                ax.plot(xs,ys,c='g')
                #for subpp in list(country_shape):
                #    if subpp.intersects(l):
                #        xs,ys = subpp.exterior.xy
                #        ax.plot(xs,ys,c='r')
                plt.show()
            if not l.is_valid:
                print (l.is_valid)
        """

        poly_shapes = polygon_shapes_from_voronoi_lines(fix_poly_lines, country_shape)
        poly_shapes = [pp for pp in poly_shapes if not pp.is_empty]

        points = coords_to_points(vor_coords)

        tree = STRtree(poly_shapes)

        ucdb_slice['exp_geom'] = ucdb_slice.apply(lambda row: self._tree_search_union(tree,row), axis=1)

        return ucdb_slice, poly_shapes, all_coords



if __name__ == "__main__":
    rc =EuclideanAllocation(
        countries_path = os.path.join(os.environ['PYTHONPATH'],'data','iso2.csv'), 
        ucdb_path = os.path.join(os.environ['PYTHONPATH'],'data','GHSL_UCDB','GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg'), 
        ne_path=os.path.join(os.environ['PYTHONPATH'],'data','ne','ne_10m_countries.gpkg') ,
        save_dir=os.path.join(os.environ['PYTHONPATH'],'data','GHSL_UCDB_EUCLID'))


    print (sorted(rc.ucdb.CTR_MN_ISO.unique()))

    for iso3 in sorted(rc.ucdb.CTR_MN_ISO.unique()):
        if iso3=='NAM':
            iso2='NA'
        else:
            iso2 = rc.iso2_df[rc.iso2_df.iso3==iso3].iloc[0].name

        if not os.path.exists(os.path.join(os.environ['PYTHONPATH'],'data','GHSL_UCDB_EUCLID',iso2+'.gpkg')):
            try:
                rc.run_country(None, str(iso3))
            except Exception as e:
                print ('ruh roh', str(iso3))
                print (e)
        else:
            print ('exists already', str(iso3))

    
    
    
    


