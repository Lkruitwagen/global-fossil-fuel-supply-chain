import os

import requests
from tqdm import tqdm
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp2d

import matplotlib.pyplot as plt


class DD:
    def __init__(self,s_year,e_year=None, heating_point=16, cooling_point = 21):
        self.heating_point=heating_point+272.15
        self.cooling_point = cooling_point+272.15

        self.s_year = s_year
        if e_year == None:
            self.e_year = self.s_year+1
        else:
            self.e_year = e_year

        self.maybe_download_climatedata()

        self.load_climatedata()

    def load_climatedata(self):

        data = {}
        
        for year in range(self.s_year, self.e_year):
            fpath = 'data/sat/air.sig995.'+str(year)+'.nc'
            nc = Dataset(fpath)
            d = np.array(nc.variables['air'])
            shp = d.shape
            d = d.reshape(4,-1,shp[-2],shp[-1]).mean(axis=0)

            data[year] = d[0:365,:,:]

            print (data[year].shape)

        dim1, dim2 = data[self.s_year].shape[-2:]

        sat = np.zeros((self.e_year -self.s_year,365,dim1,dim2))

        for ii_y, year in enumerate(range(self.s_year, self.e_year)):
            sat[ii_y,...] = data[year]

        data = None

        self.HDD = (self.heating_point - sat).clip(0.,None).sum(axis=1).mean(axis=0) # day sum then year mean 
        self.CDD = (sat - self.cooling_point).clip(0.,None).sum(axis=1).mean(axis=0) # day sum then year mean 

        self.HDD_F = interp2d(np.linspace(0,357.5,144), np.linspace(-90,90,73), self.HDD, kind='cubic')
        xx = np.linspace(0,360,360)
        yy = np.linspace(-90,90,180)

        print (self.HDD_F(29*2.5,(90-20)*2.5))
        Z = self.HDD_F(xx,yy)
        plt.imshow(Z)
        plt.show()

        #print ('HDD',self.HDD.shape, self.HDD.min(), self.HDD.max())
        plt.imshow(self.HDD)

        plt.show()
        #print ('CDD',self.CDD.shape, self.CDD.min(), self.CDD.max())
        plt.imshow(self.CDD)
        plt.show()


    def maybe_download_climatedata(self):

        for year in range(self.s_year, self.e_year):
            fpath = 'data/sat/air.sig995.'+str(year)+'.nc'
            if os.path.exists(fpath):
                print (f'Found year: {year}')
            else:
                url = 'https://www.esrl.noaa.gov/psd/thredds/fileServer/Datasets/ncep.reanalysis/surface/air.sig995.'+str(year)+'.nc'
                self._download_file(url,fpath,f'year: {str(year)}')
                

    def _download_file(self,url, filename, msg):
        """
        Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
        """
        chunkSize = 1024
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                pbar = tqdm( unit="KB", total=int( r.headers['Content-Length']), desc=msg , ncols=200)
                for chunk in r.iter_content(chunk_size=chunkSize): 
                    if chunk: # filter out keep-alive new chunks
                        pbar.update (len(chunk))
                        f.write(chunk)
            return filename
        else:
            print ('qwatch')

    #def query_HDD(self, lat,lon):


if __name__=='__main__':
    hdd = DD(2000,2003)

