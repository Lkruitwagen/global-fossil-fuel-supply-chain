import os

import requests
from tqdm import tqdm
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp2d

import matplotlib.pyplot as plt


class DD:
    def __init__(self,s_year,e_year=None, heating_point=15.5, cooling_point = 19):
        self.heating_point=heating_point+272.15
        self.cooling_point = cooling_point+272.15

        self.s_year = s_year
        if e_year == None:
            self.e_year = self.s_year+1
        else:
            self.e_year = e_year

        self.maybe_download_climatedata()

        self.load_climatedata_hourly()

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

        hdd = (self.heating_point - sat).clip(0.,None).sum(axis=1).mean(axis=0) # day sum then year mean 
        cdd = (sat - self.cooling_point).clip(0.,None).sum(axis=1).mean(axis=0) # day sum then year mean 

        HDD_F = interp2d(np.linspace(0,357.5,144), np.linspace(-90,90,73), hdd, kind='cubic')
        CDD_F = interp2d(np.linspace(0,357.5,144), np.linspace(-90,90,73), cdd, kind='cubic')
        
        xx = np.linspace(0,360,360)
        yy = np.linspace(-90,90,180)

        self.HDD = HDD_F(xx,yy)
        self.CDD = CDD_F(xx,yy)


        #print ('HDD',self.HDD.shape, self.HDD.min(), self.HDD.max())
        plt.imshow(self.HDD)

        plt.show()
        #print ('CDD',self.CDD.shape, self.CDD.min(), self.CDD.max())
        plt.imshow(self.CDD)
        plt.show()

    def load_climatedata_hourly(self):

        data = {}
        
        for year in range(self.s_year, self.e_year):
            fpath = 'data/sat/air.sig995.'+str(year)+'.nc'
            nc = Dataset(fpath)
            d = np.array(nc.variables['air'])
            
            data[year] = d[0:365*4,:,:]

        dim1, dim2 = data[self.s_year].shape[-2:]

        sat = np.zeros((self.e_year -self.s_year,365*4,dim1,dim2))

        for ii_y, year in enumerate(range(self.s_year, self.e_year)):
            sat[ii_y,...] = data[year]

        data = None

        hdd = (self.heating_point - sat).clip(0.,None).sum(axis=1).mean(axis=0)/4 # day sum then year mean 
        cdd = (sat - self.cooling_point).clip(0.,None).sum(axis=1).mean(axis=0)/4 # day sum then year mean 

        HDD_F = interp2d(np.linspace(0,357.5,144), np.linspace(-90,90,73), hdd, kind='cubic')
        CDD_F = interp2d(np.linspace(0,357.5,144), np.linspace(-90,90,73), cdd, kind='cubic')
        
        xx = np.linspace(0,360,360)
        yy = np.linspace(-90,90,180)

        self.HDD = HDD_F(xx,yy)
        self.CDD = CDD_F(xx,yy)


        #print ('HDD',self.HDD.shape, self.HDD.min(), self.HDD.max())
        #plt.imshow(self.HDD)

        #plt.show()
        #print ('CDD',self.CDD.shape, self.CDD.min(), self.CDD.max())
        #plt.imshow(self.CDD)
        #plt.show()


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

    def query_HDD(self, lat,lon):
        if lon<0:
            qlon = -1*lon+180
        else:
            qlon=lon
        return self.HDD[int(round(180-(lat+90))),int(round(qlon))]

    def query_CDD(self, lat,lon):
        if lon<0:
            qlon = -1*lon+180
        else:
            qlon=lon
        return self.CDD[int(round(180 - (lat+90))),int(round(qlon))]

    def sense_check(self):
    #sampled from degreedays.net
        test_pts = {
            'toronto':{'lat':43.67,'lon':-79.4,'HDD':3425,'CDD':305},
            'london':{'lat':51.48,'lon':0.45,'HDD':1916,'CDD':161},
            'helsinki':{'lat':60.32,'lon':24.96,'HDD':3823,'CDD':99},
            'atlanta':{'lat':33.78,'lon':-84.52,'HDD':1176,'CDD':1161},
            'san francisco':{'lat':37.62,'lon':-122.37,'HDD':879,'CDD':178},
            'las vegas':{'lat':36.08,'lon':-115.15,'HDD':905,'CDD':2003},
            'sao paolo':{'lat':-23.63,'lon':-46.66,'HDD':91,'CDD':1113},
            'beijing':{'lat':40.07,'lon':116.59,'HDD':2695,'CDD':1010},
            'johannesburg':{'lat':-26.14,'lon':28.25,'HDD':509,'CDD':584},
            'chennai':{'lat':12.99,'lon':80.18,'HDD':0,'CDD':4131},
            'wellington':{'lat':41.33,'lon':174.81,'HDD':822,'CDD':86},
            'perth':{'lat':-31.93,'lon':115.98,'HDD':484,'CDD':1093},
        }

        for kk,vv in test_pts.items():
            
            cdd = self.query_CDD(vv['lat'],vv['lon'])
            print('{name}: HDD: {hdd_true},{hdd_gen:.0f}, CDD: {cdd_true}, {cdd_gen:.0f}'.format(
                name=kk,
                hdd_true=vv['HDD'],
                hdd_gen=self.query_HDD(vv['lat'],vv['lon']),
                cdd_true=vv['CDD'],
                cdd_gen=self.query_CDD(vv['lat'],vv['lon'])))#, CDD: {self.query_CDD(vv['lat'],vv['lon'])}')



if __name__=='__main__':
    dd = DD(1960,1969)
    dd.sense_check()

