import glob, os, sys, random, logging, warnings, json

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from shapely.affinity import affine_transform
from PIL import Image, ImageDraw
from shapely import wkt, geometry
from tqdm import tqdm
tqdm.pandas()


warnings.filterwarnings('ignore')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# In this file we load the country UCDBs and allocate 

# Re-save everything down to country, so multi-country UCs just get split to two. Can recombine later if makes sense.

#

class EnergyAllocation:

    def __init__(self, iea_path,countries_path,edgar_path, ucdbs_path, ne_path,save_dir):
        logging.info(f'init...')
        self.iso2_df = pd.read_csv(countries_path).set_index('iso2')
        self.ucdbs_path = ucdbs_path
        self.save_dir = save_dir
        self.iea_path = iea_path
        iea_files = glob.glob(iea_path+'/*')
        iea_files = [f for f in iea_files if not (('WORLD' in f) or ('iso_lookup' in f))]
        self.iea_dict = {f.split('_')[-1][0:2]:f for f in iea_files}

        logging.info(f'saving to {self.save_dir}')
        self.units = {
            'GJ-per-ktoe':41868 #GJ/ktoe
        }
        self.dt_shapely = [10, 0, 0, 10, 1800, 900]
        self.edgar_paths = {
            'manufacturing':'2015_1A2_manufacturing.nc',
            'aviation_climb_descent':'2015_1A3_aviation_CDS.nc',
            'buildings':'2015_1A4_1A5_buildings.nc',
            'nonmetallic_minerals':'2015_2A_nonmetallic_minerals.nc',
            'chemical_processes':'2015_2B_chemical_processes.nc',
            'iron_steel_manufacturing':'2015_2C1_2C2_iron_steel_manufacturing.nc',
            'nonferrous_metal_production':'2015_2C3_2C4_2C5_nonferrous_metal_production.nc',
            'solvent_products':'2015_2D_solvents_products.nc',
            'road_transport':'2015_1A3b_road_transport.nc',
            'other_transport':'2015_1A3ce_other_transport.nc',
            'agricuture_soils':'2015_3C_agriculture_soils.nc',
            'nonenergy':'2015_2D_nonenergy_use.nc',
            'shipping':'1997_1A3d_shipping.nc',
        }
        self.sectors = {
            'industry':{'edgars':
                            ['manufacturing',
                            'nonmetallic_minerals',
                            'chemical_processes',
                            'iron_steel_manufacturing',
                            'nonferrous_metal_production',
                            'solvent_products'],
                        'iea_cols':'industry'},
            'transport':{'edgars':['road_transport', 'other_transport'],'iea_cols':'transport'},
            'buildings':{'edgars':['buildings'],'iea_cols':'buildings'},
            'aviation':{'edgars':['aviation_climb_descent'],'iea_cols':'international aviation bunkers****'}, # scale to full iea
            'agriculture':{'edgars':['agricuture_soils'],'iea_cols':'agriculture / forestry'}, # some industry and transport cross-contam
            'nonenergy':{'edgars':['nonenergy'],'iea_cols':'non-energy use'},
            'shipping':{'edgars':['shipping'],'iea_cols':'international marine bunkers***'} # buffer city and scale to IEA numbers
        }
        #estimate shipping by buffering port asset-level data; don't buffer entire geom

        #edgar units: kg/m^2/s

        # lower heating value, https://www.carbonfootprint.com/docs/2018_conversion_factors_2018_-_full_set__for_advanced_users__v01-00.xls
        self.emission_factors= {
            'coal':.34956*277.78/1000, #t/GJ
            'oil': .26573*277.78/1000, #t/GJ
            'gas': .20437*277.78/1000, #t/GJ
        }
        #kgCO2e/kWh * 277.78 kWh/GJ /1000 kg/t

        area_arr = np.load(os.path.join(os.environ['PYTHONPATH'],'data','edgar','area_arr.npz'))['arr']

        #all_cols: [coal,oil,gas] x [ucdb,exp_geom] x [industry, transport, buildings, aviation, shipping, agriculture, nonenergy]
        self.edgars = {}
        for kk,vv in self.edgar_paths.items():
            self.edgars[kk]=np.roll(np.array(Dataset(os.path.join(edgar_path,vv)).variables['emi_co2']),1800,axis=1)
            self.edgars[kk] = self.edgars[kk]*8760*3600/1000 *area_arr.T #t/pix/year

        """
        for kk,vv in self.edgars.items():
            fig, axs = plt.subplots(2,1,figsize=(18,18))
            axs[0].imshow(vv, origin='lower', vmin=0, vmax = 10000)
            axs[1].imshow(area_arr.T, origin='lower')
            plt.show()
        """

    def run_country(self, country_iso2, country_iso3=None, to_file=True):

        if country_iso2:
            self.iso2 = country_iso2
            self.iso3 = self.iso2_df.at[self.iso2,'iso3']
        elif country_iso3:
            self.iso3 = country_iso3
            self.iso2 = self.iso2_df[self.iso2_df.iso3==self.iso3].iloc[0].name
        logging.info(f'Running country: {self.iso2, self.iso3}')

        # What if there is no IEA??? get population portion from remaining countries and use REM.csv

        if self.iso2 not in list(self.iea_dict.keys()):
            logging.info(f'Balances not available, allocating proportionally to population.')
            self.iea_df = pd.read_csv(os.path.join(self.iea_path, 'REM.csv')).set_index('Unnamed: 0')
            self.iea_df = self.iea_df * self.iso2_df.at[self.iso2,'non_file_pop_portion']

        else:
            self.iea_df = pd.read_csv(self.iea_dict[self.iso2]).set_index('Unnamed: 0')

        self.iea_df.index = self.iea_df.index.str.lower()

        self.iea_df = self.iea_df.append(self.iea_df.loc[['commercial and public services','residential'],:].sum().rename('buildings'))

        self.ucdb_country = gpd.read_file(os.path.join(self.ucdbs_path,self.iso2+'.gpkg'))


        for kk,vv in self.sectors.items():
            for rast_key in vv['edgars']:
                logging.info(f'Doing rast key {rast_key}')
                if kk=='shipping':
                    #buffer shipping so it grabs the maritime traffic
                    self.ucdb_country['em_ed_'+rast_key] = self.ucdb_country.progress_apply(lambda row: self._sample_raster(self.edgars[rast_key],geometry.shape(json.loads(row['geom_gj'])).buffer(0.5), row['UC_NM_MN']), axis=1)
                else:
                    self.ucdb_country['em_ed_'+rast_key] = self.ucdb_country.progress_apply(lambda row: self._sample_raster(self.edgars[rast_key],row['geometry'], row['UC_NM_MN']), axis=1)

            sec_cols = ['em_ed_'+c for c in vv['edgars']]
            self.ucdb_country['em_sec_'+kk] = self.ucdb_country.loc[:,sec_cols].sum(axis=1)

            #t/yr
            em_sec_coal = self.iea_df.at[vv['iea_cols'],'Coal*']*self.units['GJ-per-ktoe']*self.emission_factors['coal']
            em_sec_oil = self.iea_df.at[vv['iea_cols'],'Oil products']*self.units['GJ-per-ktoe']*self.emission_factors['oil']
            em_sec_gas = self.iea_df.at[vv['iea_cols'],'Natural gas']*self.units['GJ-per-ktoe']*self.emission_factors['gas']

            em_sec_total = em_sec_coal + em_sec_oil + em_sec_gas

            por_sec_coal = em_sec_coal / em_sec_total
            por_sec_oil = em_sec_oil / em_sec_total
            por_sec_gas = em_sec_gas / em_sec_total

            # GJ/yr
            self.ucdb_country['en_sec_'+kk+'_coal'] = self.ucdb_country['em_sec_'+kk] * por_sec_coal / self.emission_factors['coal']
            self.ucdb_country['en_sec_'+kk+'_oil'] = self.ucdb_country['em_sec_'+kk] * por_sec_oil / self.emission_factors['oil']
            self.ucdb_country['en_sec_'+kk+'_gas'] = self.ucdb_country['em_sec_'+kk] * por_sec_gas / self.emission_factors['gas']
        
        # normalise these to the IEA
        for kk in ['aviation', 'shipping']:
            self.ucdb_country['en_sec_'+kk+'_coal'] = (self.ucdb_country['en_sec_'+kk+'_coal'] / self.ucdb_country['en_sec_'+kk+'_coal'].sum()).fillna(0) * -1 * self.iea_df.at[self.sectors[kk]['iea_cols'],'Coal*'] * self.units['GJ-per-ktoe']
            self.ucdb_country['en_sec_'+kk+'_oil'] = (self.ucdb_country['en_sec_'+kk+'_oil'] / self.ucdb_country['en_sec_'+kk+'_oil'].sum()).fillna(0) * -1 *self.iea_df.at[self.sectors[kk]['iea_cols'],'Oil products'] * self.units['GJ-per-ktoe']
            self.ucdb_country['en_sec_'+kk+'_gas'] = (self.ucdb_country['en_sec_'+kk+'_gas'] / self.ucdb_country['en_sec_'+kk+'_gas'].sum()).fillna(0) * -1 *self.iea_df.at[self.sectors[kk]['iea_cols'],'Natural gas'] * self.units['GJ-per-ktoe']
        
        self.ucdb_country = self.ucdb_country.fillna(0)

        if to_file:
            self.ucdb_country.to_file(os.path.join(self.save_dir,self.iso2+'.gpkg'),driver='GPKG')

    def _sample_raster(self,raster,rowshp,name):

        im = Image.fromarray(np.zeros(raster.shape), mode='L')
        draw = ImageDraw.Draw(im)

        shp = wkt.loads(wkt.dumps(rowshp, rounding_precision=1))
        pix_shp = affine_transform(shp.buffer(0), self.dt_shapely)
            
        if pix_shp.type=='MultiPolygon' and not pix_shp.is_empty:
            for subshp in list(pix_shp):
                lons, lats = subshp.exterior.xy
                draw.polygon(list(zip(lons, lats)), fill=255)
        elif pix_shp.type=='Polygon' and not pix_shp.is_empty:
            lons, lats = pix_shp.exterior.xy
            draw.polygon(list(zip(lons, lats)), fill=255)

        mask = np.array(im)

        #logging.info(f'{name}, {np.sum(raster*(mask>0))}')

        return np.sum(raster*(mask>0))

    def visualise_country(self, to_file=True, showfig=False):
        fig, axs = plt.subplots(len(self.sectors.keys()),3,figsize=(len(self.sectors.keys())*6,18))

        for ii_k,kk in enumerate(self.sectors.keys()):
            for ii_vec, vec in enumerate(['coal','oil','gas']):
                #print (self.ucdb_country['en_sec_'+kk+'_'+vec])
                #print (self.ucdb_country.loc[:,'en_sec_'+kk+'_'+vec])
                self.ucdb_country.plot(column='en_sec_'+kk+'_'+vec, ax=axs[ii_k,ii_vec])

                if ii_vec !=0:
                    axs[ii_k,ii_vec].axis('off')
                else:
                    # make xaxis invisibel
                    axs[ii_k,ii_vec].xaxis.set_visible(False)
                    # make spines (the box) invisible
                    plt.setp(axs[ii_k,ii_vec].spines.values(), visible=False)
                    # remove ticks and labels for the left axis
                    axs[ii_k,ii_vec].tick_params(left=False, labelleft=False)
                    #remove background patch (only needed for non-white background)
                    #axs[ii_k,ii_vec].patch.set_visible(False)

        for ii_vec, vec in enumerate(['coal','oil','gas']):
            axs[0,ii_vec].set_title(vec)

        for ii_k,kk in enumerate(self.sectors.keys()):
            axs[ii_k,0].set_ylabel(kk)

        if to_file:
            plt.savefig(os.path.join(self.save_dir,self.iso2+'.png'))

        if showfig:
            plt.show()




if __name__ == "__main__":
    ea = EnergyAllocation(
            iea_path=os.path.join(os.environ['PYTHONPATH'],'data','iea_balances_public'),
            countries_path=os.path.join(os.environ['PYTHONPATH'],'data','iso2.csv'),
            edgar_path=os.path.join(os.environ['PYTHONPATH'],'data','edgar'), 
            ucdbs_path=os.path.join(os.environ['PYTHONPATH'],'data','GHSL_UCDB_EUCLID'), 
            ne_path=os.path.join(os.environ['PYTHONPATH'],'data','ne'),
            save_dir=os.path.join(os.environ['PYTHONPATH'],'data','GHSL_UCDB_ENERGY'))

    all_countries = sorted(
                        list(
                            set(
                                [f.split('/')[-1][0:2] for f in glob.glob(os.path.join(os.environ['PYTHONPATH'],'data','GHSL_UCDB_EUCLID','*'))])))

    for iso2 in all_countries:

        if not os.path.exists(os.path.join(os.environ['PYTHONPATH'],'data','GHSL_UCDB_ENERGY',iso2+'.gpkg')):
            try:
                ea.run_country(iso2)
                ea.visualise_country()
            except Exception as e:
                print ('ruh roh', str(iso2))
                print (e)
        else:
            print ('exists already', str(iso2))