import logging, os, sys, pickle, json

import pandas as pd
from math import pi
import numpy as np

import geopandas as gpd

logger=logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def prep_oilwells(df, iso2):
    df.loc[[2,3],'md_country'] = 'Congo, (Brazzaville)'
    df.loc[[7,8],'md_country'] = "Cote d'Ivoire"
    df.loc[1863,'md_country'] = 'iran, islamic republic of'
    
    df['md_country'] = df['md_country'].str.lower()
    iso2['country'] = iso2['country'].str.lower()
    
    df = pd.merge(df, iso2[['country','iso2']], how='left',left_on='md_country',right_on='country')
    df['unique_id'] = 'OILWELL_' + df.index.astype(str)

    return df[['unique_id','iso2']]

def prep_coalmines(df, iso2, ne):

    df['md_country'] = df['md_country'].str.lower()
    iso2['country'] = iso2['country'].str.lower()
    df = pd.merge(df, iso2[['iso2','iso3']], how='left',left_on='iso_a3',right_on='iso3' )
    df = pd.merge(df, iso2[['iso2','country']], how='left',left_on='md_country',right_on='country' )
    df['iso2'] = df['iso2_x'].fillna(df['iso2_y'])

    df.loc[df['md_country']=='Iran (Islamic Republic of','iso2']='IR'
    df.loc[df['md_country']=='United Republic of Tanzani','iso2']='TZ'
    
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df = gpd.sjoin(gpd.GeoDataFrame(df, geometry='geometry'), ne[['ISO_A2','geometry']], how='left')
    df['iso2'] = df['iso2'].fillna(df['ISO_A2'])
    print (df['iso2'])
    print (df['iso2'].isna().sum())
    df['unique_id'] = 'COALMINE_'+df.index.astype(str)
    
    return df[['unique_id','iso2']]

def prep_oilfields(df_oilfields, iso2):
    df_oilfields['md_country'] = df_oilfields['md_country'].str.split(';')
    df = pd.DataFrame(df_oilfields).reset_index().explode('md_country')[['index','md_country']]
    df['md_country'] = df['md_country'].str.lower().str.strip()
    iso2['country'] = iso2['country'].str.lower()
    
    df = pd.merge(df, iso2[['country','iso2']], how='left',left_on='md_country', right_on='country')
    
    df.loc[df['md_country'].str.contains('venezuela'),'iso2'] = 'VE'
    df.loc[df['md_country'].str.contains('cote d'),'iso2'] = 'CI'
    df.loc[df['md_country'].str.contains('iran'),'iso2'] = 'IR'
    df.loc[df['md_country'].str.contains('syria'),'iso2'] = 'SY'
    df.loc[df['md_country'].str.contains('democratic republic of the congo'),'iso2'] = 'CD'
    df.loc[df['md_country'].str.contains('congo'),'iso2'] = 'CG'
    df.loc[df['md_country'].str.contains('republic of korea'),'iso2'] = 'KR'
    df.loc[df['md_country'].str.contains('s republic of korea'),'iso2'] = 'KP'
    df.loc[df['md_country'].str.contains('tanzania'),'iso2'] = 'TZ'
    df.loc[df['md_country'].str.contains('guinea bissau'),'iso2'] = 'GW'
    df.loc[df['md_country'].str.contains('bolivia'),'iso2'] = 'BO'
    df.loc[df['md_country'].str.contains('moldova'),'iso2'] = 'MD'
    df.loc[df['md_country'].str.contains('macedonia'),'iso2'] = 'MK'
    df.loc[df['md_country'].str.contains('lao'),'iso2'] = 'LA'
    
    df = pd.DataFrame(df.groupby('index')['iso2'].apply(list))
    
    df['iso2'] = df['iso2'].apply(lambda el: ';'.join(el))
    
    df['unique_id'] = 'OILFIELD_' + df.index.astype(str)
    
    return df

def prep_cities(df_cities):
    
    carrier_columns = {carrier:[f'en_sec_{sector}_{carrier}' for sector in ['industry','transport','buildings','aviation','agriculture','nonenergy','shipping']] for carrier in ['coal','oil','gas']}
    
    for carrier in ['coal','oil','gas']:
        df_cities[f'total_{carrier}_consumption'] = df_cities[carrier_columns[carrier]].fillna(0).sum(axis=1)
    
    logger.info('Doing Features column:')
    df_cities['features'] = df_cities[[f'total_{carrier}_consumption' for carrier in ['coal','oil','gas']]].to_dict(orient='records')
    df_cities['features'] = df_cities['features'].progress_apply(lambda el: json.dumps(el))
    
    df_cities['unique_id'] = 'CITY_'+df_cities.reset_index()['index'].astype(str)
    
    return pd.DataFrame(df_cities)[['unique_id','features']]