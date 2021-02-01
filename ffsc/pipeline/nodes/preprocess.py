import logging, sys, json
#import geopandas as gpd

import pandas as pd
from shapely import geometry
from tqdm import tqdm
tqdm.pandas()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def do_preprocess(gdf, idx_column, idx_prefix, feature_columns, logger):
    
    gdf = pd.DataFrame(gdf).reset_index().rename(columns={idx_column:'unique_id'})
    
    logger.info('Doing geometry to str:')
    gdf['geometry'] = gdf['geometry'].progress_apply(lambda el: el.wkt)
    
    if not feature_columns:
        gdf['features'] = [json.dumps(dict())]*len(gdf)
    else:
        gdf['features'] = gdf[feature_columns].to_dict(orient='records')
        logger.info('Doing Features column:')
        gdf['features'] = gdf['features'].progress_apply(lambda el: json.dumps(el))
    
    gdf['unique_id'] = idx_prefix + '_' + gdf['unique_id'].astype(str)
    
    return gdf[['unique_id','features','geometry']]
    

def preprocess_shippingroutes(gdf):
    logger = logging.getLogger('preprocess_shippingroutes')
    return do_preprocess(gdf, 'index','SHIPPINGROUTE',None, logger)

def preprocess_ports(gdf):
    logger = logging.getLogger('preprocess_ports')
    return do_preprocess(gdf, 'index','PORT',None, logger)

def preprocess_pipelines(fc):
    logger = logging.getLogger('preprocess_pipelines')
    logger.info(f'Len fts: {len(fc["features"])}')
    
    records = []
    for ii_f, ft in enumerate(tqdm(fc['features'])):
        records.append(dict(
                unique_id='PIPELINE_'+str(ii_f),
                features=json.dumps({}),
                geometry=geometry.shape(ft['geometry']).wkt
            )
        )    
    
    return pd.DataFrame.from_records(records)

def preprocess_coalmines(gdf):
    logger = logging.getLogger('preprocess_coalmines')

    return do_preprocess(gdf, 'index','COALMINE',None, logger)

def preprocess_oilfields(gdf):
    logger = logging.getLogger('preprocess_oilfields')

    return do_preprocess(gdf, 'index','OILFIELD',None, logger)

def preprocess_lngterminals(gdf):
    logger = logging.getLogger('preprocess_lngterminals')

    return do_preprocess(gdf, 'index','LNGTERMINAL',None, logger)

def preprocess_powerstations(gdf):
    logger = logging.getLogger('preprocess_powerstations')
    gdf = gdf[gdf[['fuel1','fuel2','fuel3','fuel4']].isin(['Gas','Oil','Coal']).any(axis=1)]
    gdf = gdf[~((gdf['latitude']>90) | (gdf['latitude']<-90) | (gdf['longitude']<-180) | (gdf['longitude']>180))]
    
    return do_preprocess(gdf, 'index','POWERSTATION',['capacity_mw','fuel1','fuel2','fuel3','fuel4'], logger)

def preprocess_railways(fc):
    logger = logging.getLogger('preprocess_railways')
    logger.info(f'Len fts: {len(fc["features"])}')
    
    records = []
    for ii_f, ft in enumerate(tqdm(fc['features'])):
        records.append(dict(
                unique_id='RAILWAY_'+str(ii_f),
                features=json.dumps({}),
                geometry=geometry.shape(ft['geometry']).wkt
            )
        )    
    
    return pd.DataFrame.from_records(records)

def preprocess_refineries(gdf_refineries,gdf_processingplants):
    logger = logging.getLogger('preprocess_refineries')
    gdf = pd.concat([gdf_refineries,gdf_processingplants])
    gdf['new_index'] = range(len(gdf))

    return do_preprocess(gdf, 'new_index','REFINERY',None, logger)

def preprocess_oilwells(gdf):
    logger = logging.getLogger('preprocess_oilwells')

    return do_preprocess(gdf, 'index','OILWELL',None, logger)

def preprocess_cities_base(gdf):
    logger = logging.getLogger('preprocess_cities')
    
    gdf = pd.DataFrame(gdf).reset_index().rename(columns={'index':'unique_id'})
    gdf['unique_id'] = 'CITY' + '_' + gdf['unique_id'].astype(str)
    
    logger.info('Doing geometry to str:')
    gdf['geometry'] = gdf['geom_gj'].progress_apply(lambda el: geometry.shape(el).wkt)
    #gdf = gdf.rename(columns={'geom_gj':'features'})
    
    #logger.info('Doing small geometry to str:')
    gdf['features'] = [json.dumps(dict())]*len(gdf)#gdf['features'].progress_apply(lambda el: geometry.shape(el).wkt)

    return gdf[['unique_id','geometry','features']]

def preprocess_cities_euclid(gdf):
    logger = logging.getLogger('preprocess_cities')
    
    gdf = pd.DataFrame(gdf).reset_index().rename(columns={'index':'unique_id'})
    gdf['unique_id'] = 'CITY' + '_' + gdf['unique_id'].astype(str)
    
    logger.info('Doing geometry to str:')
    gdf['geometry'] = gdf['geometry'].progress_apply(lambda el: el.wkt)
    gdf = gdf.rename(columns={'geom_gj':'features'})
    
    logger.info('Doing small geometry to str:')
    gdf['features'] = gdf['features'].progress_apply(lambda el: geometry.shape(el).wkt)

    return gdf