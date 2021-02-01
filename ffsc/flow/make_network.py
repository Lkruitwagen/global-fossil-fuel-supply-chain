import logging, json, os, sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from ffsc.flow.recipes import recipes

def get_edges(df,dup_1, dup_2, reverse):
    edges = ['START','END']
    if dup_1:
        edges[0] = 'START_B'
    if dup_2:
        edges[1] = 'END_B'
    if reverse:
        edges = [edges[1], edges[0]]

    return df[edges+['IMPEDANCE']].to_records(index=False).tolist()

def make_coal_network(
    df_cities, 
    df_powerstations,
    df_coalmines,
    df_edges_cities,
    df_edges_powerstations,
    df_edges_coalmines,
    df_edges_other_railways,
    df_edges_railways_other,
    df_edges_shippingroutes_other,
    df_edges_railways_railways,
    df_edges_shippingroutes_shippingroutes, flow_parameters):
    
    logger = logging.getLogger(f'flow_edges_coal')
    
    for df in [df_cities, 
                df_powerstations,
                df_coalmines,
                df_edges_cities,
                df_edges_powerstations,
                df_edges_coalmines,
                df_edges_other_railways,
                df_edges_railways_other,
                df_edges_shippingroutes_other,
                df_edges_railways_railways,
                df_edges_shippingroutes_shippingroutes]:
        print (df.head(5))
        
    edge_dfs = [df_edges_cities,
                df_edges_powerstations,
                df_edges_coalmines,
                df_edges_other_railways,
                df_edges_railways_other,
                df_edges_shippingroutes_other,
                df_edges_railways_railways,
                df_edges_shippingroutes_shippingroutes]
    
    names = ['df_edges_cities',
            'df_edges_powerstations',
            'df_edges_coalmines',
            'df_edges_other_railways',
            'df_edges_railways_other',
            'df_edges_shippingroutes_other',
            'df_edges_other_shippingroutes',
            'df_edges_railways_railways',
            'df_edges_shippingroutes_shippingroutes']
        
    for df, name in zip(edge_dfs, names):
        logger.info(f'{name}, {df["START"].str.split("_").str[0].unique()},  {df["END"].str.split("_").str[0].unique()}')
        
    ## trim for coal
    logger.info('Trimming for coal')
    powerstations_noncoal = df_powerstations.loc[~df_powerstations['features'].apply(lambda el: json.loads(el)['fuel1']=='Coal'),'unique_id'].values
    df_powerstations = df_powerstations[~df_powerstations['unique_id'].isin(powerstations_noncoal)]
    df_edges_powerstations = df_edges_powerstations[df_edges_powerstations['END'].isin(df_powerstations['unique_id'].values)]
    df_edges_railways_other = df_edges_railways_other[~df_edges_railways_other['END'].isin(powerstations_noncoal)]
    df_edges_shippingroutes_other = df_edges_shippingroutes_other[~df_edges_shippingroutes_other['END'].str.split('_').str[0].isin(['LNGTERMINAL','SHIPPINGROUTE'])]

    ### get ingredients
    df_ingredients = {
        'coalmines-railways':df_edges_other_railways.copy(),
        'coalmines-firstmile':df_edges_coalmines.copy(),
        'railways-railways':df_edges_railways_railways.copy(),
        'railways-ports':df_edges_railways_other[df_edges_railways_other['END'].str.split('_').str[0]=='PORT'].copy(),
        'shipping-ports':df_edges_shippingroutes_other[df_edges_shippingroutes_other['END'].str.split('_').str[0]=='PORT'].copy(),
        'shipping-shipping':df_edges_shippingroutes_shippingroutes.copy(),
        'railways-powerstations':df_edges_railways_other[df_edges_railways_other['END'].str.split('_').str[0]=='POWERSTATION'].copy(),
        'railways-cities':df_edges_railways_other[df_edges_railways_other['END'].str.split('_').str[0]=='CITY'].copy(),
        'lastmile-powerstations': df_edges_powerstations.copy(),
        'cities-lastmile':df_edges_cities.copy()
    }
    
    ### add impendances
    logger.info('Adding impedances')
    df_ingredients['coalmines-railways']['IMPEDANCE']= (df_ingredients['coalmines-railways']['DISTANCE']/1000*flow_parameters['RAILCOST'] + flow_parameters['RAILLOAD']/2)*flow_parameters['tperTJ']['coal']
    df_ingredients['coalmines-firstmile']['IMPEDANCE']=(df_ingredients['coalmines-firstmile']['DISTANCE']/1000*flow_parameters['ROADCOST'] + flow_parameters['ROADLOAD']/2)*flow_parameters['tperTJ']['coal']
    df_ingredients['railways-railways']['IMPEDANCE']=(df_ingredients['railways-railways']['DISTANCE']/1000*flow_parameters['RAILCOST'])*flow_parameters['tperTJ']['coal']
    df_ingredients['railways-ports']['IMPEDANCE']=(df_ingredients['railways-ports']['DISTANCE']/1000*flow_parameters['RAILCOST']+ flow_parameters['RAILLOAD']/2)*flow_parameters['tperTJ']['coal']
    df_ingredients['shipping-ports']['IMPEDANCE']=(df_ingredients['shipping-ports']['DISTANCE']/1000*flow_parameters['SEACOST'] + flow_parameters['SEALOAD']/2)*flow_parameters['tperTJ']['coal']
    df_ingredients['shipping-shipping']['IMPEDANCE']=(df_ingredients['shipping-shipping']['DISTANCE']/1000*flow_parameters['SEACOST'])*flow_parameters['tperTJ']['coal']
    df_ingredients['railways-powerstations']['IMPEDANCE']=(df_ingredients['railways-powerstations']['DISTANCE']/1000*flow_parameters['RAILCOST'] + flow_parameters['RAILLOAD']/2)*flow_parameters['tperTJ']['coal']
    df_ingredients['railways-cities']['IMPEDANCE']=(df_ingredients['railways-cities']['DISTANCE']/1000*flow_parameters['RAILCOST'] + flow_parameters['RAILLOAD']/2)*flow_parameters['tperTJ']['coal']
    df_ingredients['lastmile-powerstations']['IMPEDANCE']=(df_ingredients['lastmile-powerstations']['DISTANCE']/1000*flow_parameters['ROADCOST'] + flow_parameters['ROADLOAD']/2)*flow_parameters['tperTJ']['coal']
    df_ingredients['cities-lastmile']['IMPEDANCE']=(df_ingredients['cities-lastmile']['DISTANCE']/1000*flow_parameters['ROADCOST'])*flow_parameters['tperTJ']['coal']
    
    for step in recipes['coal']:
        if step['dup_1']==True:
            df_ingredients[step['name']]['START_B'] = df_ingredients[step['name']]['START']+'_B'
        if step['dup_2']==True:
            df_ingredients[step['name']]['END_B'] = df_ingredients[step['name']]['END'] +'_B'
    
    ### assemble
    logger.info('assembling edge dataframe')
    all_edges = []
    for step in recipes['coal']:
        all_edges += get_edges(df_ingredients[step['name']], step['dup_1'], step['dup_2'], step['reverse'])
        
    print (len(all_edges))
        
    return pd.DataFrame(all_edges, columns=['START','END','IMPEDANCE'])

def make_oil_network(
    df_cities, 
    df_powerstations,
    df_oilfields,
    df_oilwells,
    df_edges_cities,
    df_edges_powerstations,
    df_edges_oilfields,
    df_edges_oilwells,
    df_edges_other_pipelines,
    df_edges_pipelines_other,
    df_edges_pipelines_pipelines,
    df_edges_shippingroutes_other,
    df_edges_shippingroutes_shippingroutes, 
    flow_parameters):
    

    
    logger = logging.getLogger(f'flow_edges_oil')
        
    edge_dfs = [df_edges_cities,
                df_edges_powerstations,
                df_edges_oilfields,
                df_edges_oilwells,
                df_edges_other_pipelines,
                df_edges_pipelines_other,
                df_edges_pipelines_pipelines,
                df_edges_shippingroutes_other,
                df_edges_shippingroutes_shippingroutes]
    
    names = ['df_edges_cities',
            'df_edges_powerstations',
            'df_edges_oilfields',
            'df_edges_oilwells',
            'df_edges_other_pipelines',
            'df_edges_pipelines_other',
            'df_edges_pipelines_pipelines',
            'df_edges_shippingroutes_other',
            'df_edges_shippingroutes_shippingroutes']
        
    for df, name in zip(edge_dfs, names):
        logger.info(f'{name}, {df["START"].str.split("_").str[0].unique()},  {df["END"].str.split("_").str[0].unique()}')
        
    ## trim for oil
    logger.info('Trimming for oil')
    print ('step 1')
    print (df_edges_powerstations)
    powerstations_nonoil = df_powerstations.loc[~df_powerstations['features'].apply(lambda el: json.loads(el)['fuel1']=='Oil'),'unique_id'].values
    df_powerstations = df_powerstations[~df_powerstations['unique_id'].isin(powerstations_nonoil)]
    print ('step 2')
    print (df_edges_powerstations)
    df_edges_powerstations = df_edges_powerstations[df_edges_powerstations['END'].isin(df_powerstations['unique_id'].values)]
    print ('step 3')
    print (df_edges_powerstations)
    df_edges_pipelines_other = df_edges_pipelines_other[~df_edges_pipelines_other['END'].isin(powerstations_nonoil)]
    #print (df_edges_pipelines_other)
    #print (df_edges_pipelines_other['END'].str.split('_').str[0]=='LNGTERMINAL')
    df_edges_pipelines_other = df_edges_pipelines_other[~(df_edges_pipelines_other['END'].str.split('_').str[0]=='LNGTERMINAL')]
    df_edges_shippingroutes_other = df_edges_shippingroutes_other[~(df_edges_shippingroutes_other['END'].str.split('_').str[0]=='LNGTERMINAL')]
    
    
    ### get ingredients
    df_ingredients = {
        'pipelines-oilfields':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='OILFIELD'].copy(),
        'pipelines-oilwells':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='OILWELL'].copy(),
        'oilfields-firstmile':df_edges_other_pipelines[df_edges_other_pipelines['START'].str.split('_').str[0]=='OILFIELD'].copy(),
        'oilwells-firstmile':df_edges_other_pipelines[df_edges_other_pipelines['START'].str.split('_').str[0]=='OILWELL'].copy(),
        'pipelines-ports':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='PORT'].copy(),
        'pipelines-refineries':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='REFINERY'].copy(),
        'shipping-ports':df_edges_shippingroutes_other[df_edges_shippingroutes_other['END'].str.split('_').str[0]=='PORT'].copy(),
        'shipping-shipping':df_edges_shippingroutes_shippingroutes.copy(),
        'pipelines-pipelines':df_edges_pipelines_pipelines.copy(),
        'pipelines-cities':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='CITY'].copy(),
        'pipelines-powerstations':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='POWERSTATION'].copy(),
        'lmports-powerstations': df_edges_powerstations[df_edges_powerstations['START'].str.split('_').str[0]=='PORT'].copy(),
        'lmcities-powerstations': df_edges_powerstations[df_edges_powerstations['START'].str.split('_').str[0]=='CITY'].copy(),
        'cities-lastmile':df_edges_cities.copy()
    }
    

    
    ### add impendances
    logger.info('Adding impedances')
    df_ingredients['pipelines-oilfields']['IMPEDANCE']    = (df_ingredients['pipelines-oilfields']['DISTANCE']/1000*flow_parameters['OIL_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-oilwells']['IMPEDANCE']     = (df_ingredients['pipelines-oilwells']['DISTANCE']/1000*flow_parameters['OIL_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['oilfields-firstmile']['IMPEDANCE']    = (df_ingredients['oilfields-firstmile']['DISTANCE']/1000*flow_parameters['ROADCOST'])*flow_parameters['tperTJ']['oil']
    df_ingredients['oilwells-firstmile']['IMPEDANCE']     = (df_ingredients['oilwells-firstmile']['DISTANCE']/1000*flow_parameters['ROADCOST'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-ports']['IMPEDANCE']        = (df_ingredients['pipelines-ports']['DISTANCE']/1000*flow_parameters['OIL_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-refineries']['IMPEDANCE']   = (df_ingredients['pipelines-refineries']['DISTANCE']/1000*flow_parameters['OIL_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['shipping-ports']['IMPEDANCE']         = (df_ingredients['shipping-ports']['DISTANCE']/1000*flow_parameters['SEACOST'] + flow_parameters['SEALOAD']/2)*flow_parameters['tperTJ']['oil']
    df_ingredients['shipping-shipping']['IMPEDANCE']      = (df_ingredients['shipping-shipping']['DISTANCE']/1000*flow_parameters['SEACOST'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-pipelines']['IMPEDANCE']    = (df_ingredients['pipelines-pipelines']['DISTANCE']/1000*flow_parameters['OIL_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-cities']['IMPEDANCE']       = (df_ingredients['pipelines-cities']['DISTANCE']/1000*flow_parameters['OIL_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-powerstations']['IMPEDANCE']= (df_ingredients['pipelines-powerstations']['DISTANCE']/1000*flow_parameters['OIL_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['lmcities-powerstations']['IMPEDANCE'] = (df_ingredients['lmcities-powerstations']['DISTANCE']/1000*flow_parameters['ROADCOST'])*flow_parameters['tperTJ']['oil']
    df_ingredients['lmports-powerstations']['IMPEDANCE']  = (df_ingredients['lmports-powerstations']['DISTANCE']/1000*flow_parameters['ROADCOST'])*flow_parameters['tperTJ']['oil']
    df_ingredients['cities-lastmile']['IMPEDANCE']        = (df_ingredients['cities-lastmile']['DISTANCE']/1000*flow_parameters['ROADCOST'])*flow_parameters['tperTJ']['oil']
    
    for step in recipes['oil']:
        if step['dup_1']==True:
            df_ingredients[step['name']]['START_B'] = df_ingredients[step['name']]['START']+'_B'
        if step['dup_2']==True:
            df_ingredients[step['name']]['END_B'] = df_ingredients[step['name']]['END'] +'_B'
    
    ### assemble
    logger.info('assembling edge dataframe')
    all_edges = []
    for step in recipes['oil']:
        all_edges += get_edges(df_ingredients[step['name']], step['dup_1'], step['dup_2'], step['reverse'])
        
    print (len(all_edges))
        
    return pd.DataFrame(all_edges, columns=['START','END','IMPEDANCE'])


def make_gas_network(
    df_cities, 
    df_powerstations,
    df_oilfields,
    df_oilwells,
    df_edges_cities,
    df_edges_powerstations,
    df_edges_oilfields,
    df_edges_oilwells,
    df_edges_other_pipelines,
    df_edges_pipelines_other,
    df_edges_pipelines_pipelines,
    df_edges_shippingroutes_other,
    df_edges_shippingroutes_shippingroutes, 
    flow_parameters):
    

    
    logger = logging.getLogger(f'flow_edges_Gas')
        
    edge_dfs = [df_edges_cities,
                df_edges_powerstations,
                df_edges_oilfields,
                df_edges_oilwells,
                df_edges_other_pipelines,
                df_edges_pipelines_other,
                df_edges_pipelines_pipelines,
                df_edges_shippingroutes_other,
                df_edges_shippingroutes_shippingroutes]
    
    names = ['df_edges_cities',
            'df_edges_powerstations',
            'df_edges_oilfields',
            'df_edges_oilwells',
            'df_edges_other_pipelines',
            'df_edges_pipelines_other',
            'df_edges_pipelines_pipelines',
            'df_edges_shippingroutes_other',
            'df_edges_shippingroutes_shippingroutes']
        
    for df, name in zip(edge_dfs, names):
        logger.info(f'{name}, {df["START"].str.split("_").str[0].unique()},  {df["END"].str.split("_").str[0].unique()}')
        
    ## trim for oil
    logger.info('Trimming for gas')
    powerstations_nonoil = df_powerstations.loc[~df_powerstations['features'].apply(lambda el: json.loads(el)['fuel1']=='Gas'),'unique_id'].values
    df_powerstations = df_powerstations[~df_powerstations['unique_id'].isin(powerstations_nonoil)]
    df_edges_powerstations = df_edges_powerstations[df_edges_powerstations['END'].isin(df_powerstations['unique_id'].values)]
    df_edges_pipelines_other = df_edges_pipelines_other[~df_edges_pipelines_other['END'].isin(powerstations_nonoil)]
    df_edges_pipelines_other = df_edges_pipelines_other[~(df_edges_pipelines_other['END'].str.split('_').str[0]=='PORT')]
    df_edges_shippingroutes_other = df_edges_shippingroutes_other[~(df_edges_shippingroutes_other['END'].str.split('_').str[0]=='PORT')]
    
    
    ### get ingredients
    df_ingredients = {
        'pipelines-oilfields':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='OILFIELD'].copy(),
        'pipelines-oilwells':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='OILWELL'].copy(),
        'oilfields-firstmile':df_edges_other_pipelines[df_edges_other_pipelines['START'].str.split('_').str[0]=='OILFIELD'].copy(),
        'oilwells-firstmile':df_edges_other_pipelines[df_edges_other_pipelines['START'].str.split('_').str[0]=='OILWELL'].copy(),
        'pipelines-lng':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='LNGTERMINAL'].copy(),
        'pipelines-refineries':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='REFINERY'].copy(),
        'shipping-lng':df_edges_shippingroutes_other[df_edges_shippingroutes_other['END'].str.split('_').str[0]=='LNGTERMINAL'].copy(),
        'shipping-shipping':df_edges_shippingroutes_shippingroutes.copy(),
        'pipelines-pipelines':df_edges_pipelines_pipelines.copy(),
        'pipelines-cities':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='CITY'].copy(),
        'pipelines-powerstations':df_edges_pipelines_other[df_edges_pipelines_other['END'].str.split('_').str[0]=='POWERSTATION'].copy(),
        'lmports-powerstations': df_edges_powerstations[df_edges_powerstations['START'].str.split('_').str[0]=='PORT'].copy(),
        'lmcities-powerstations': df_edges_powerstations[df_edges_powerstations['START'].str.split('_').str[0]=='CITY'].copy(),
        'cities-lastmile':df_edges_cities.copy()
    }
    

    
    ### add impendances
    logger.info('Adding impedances')
    df_ingredients['pipelines-oilfields']['IMPEDANCE']    = (df_ingredients['pipelines-oilfields']['DISTANCE']/1000*flow_parameters['GAS_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-oilwells']['IMPEDANCE']     = (df_ingredients['pipelines-oilwells']['DISTANCE']/1000*flow_parameters['GAS_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['oilfields-firstmile']['IMPEDANCE']    = (df_ingredients['oilfields-firstmile']['DISTANCE']/1000*flow_parameters['ROADCOST'])*flow_parameters['tperTJ']['oil']
    df_ingredients['oilwells-firstmile']['IMPEDANCE']     = (df_ingredients['oilwells-firstmile']['DISTANCE']/1000*flow_parameters['ROADCOST'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-lng']['IMPEDANCE']          = (df_ingredients['pipelines-lng']['DISTANCE']/1000*flow_parameters['GAS_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-refineries']['IMPEDANCE']   = (df_ingredients['pipelines-refineries']['DISTANCE']/1000*flow_parameters['GAS_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['shipping-lng']['IMPEDANCE']           = (df_ingredients['shipping-lng']['DISTANCE']/1000*flow_parameters['SEACOST'] + flow_parameters['SEALOAD']/2)*flow_parameters['tperTJ']['oil']
    df_ingredients['shipping-shipping']['IMPEDANCE']      = (df_ingredients['shipping-shipping']['DISTANCE']/1000*flow_parameters['SEACOST'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-pipelines']['IMPEDANCE']    = (df_ingredients['pipelines-pipelines']['DISTANCE']/1000*flow_parameters['GAS_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-cities']['IMPEDANCE']       = (df_ingredients['pipelines-cities']['DISTANCE']/1000*flow_parameters['GAS_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['pipelines-powerstations']['IMPEDANCE']= (df_ingredients['pipelines-powerstations']['DISTANCE']/1000*flow_parameters['GAS_PIPELINE'])*flow_parameters['tperTJ']['oil']
    df_ingredients['lmcities-powerstations']['IMPEDANCE'] = (df_ingredients['lmcities-powerstations']['DISTANCE']/1000*flow_parameters['ROADCOST'])*flow_parameters['tperTJ']['oil']
    df_ingredients['lmports-powerstations']['IMPEDANCE']  = (df_ingredients['lmports-powerstations']['DISTANCE']/1000*flow_parameters['ROADCOST'])*flow_parameters['tperTJ']['oil']
    df_ingredients['cities-lastmile']['IMPEDANCE']        = (df_ingredients['cities-lastmile']['DISTANCE']/1000*flow_parameters['ROADCOST'])*flow_parameters['tperTJ']['oil']
    
    for step in recipes['gas']:
        if step['dup_1']==True:
            df_ingredients[step['name']]['START_B'] = df_ingredients[step['name']]['START']+'_B'
        if step['dup_2']==True:
            df_ingredients[step['name']]['END_B'] = df_ingredients[step['name']]['END'] +'_B'
    
    ### assemble
    logger.info('assembling edge dataframe')
    all_edges = []
    for step in recipes['gas']:
        all_edges += get_edges(df_ingredients[step['name']], step['dup_1'], step['dup_2'], step['reverse'])
        
    print (len(all_edges))
        
    return pd.DataFrame(all_edges, columns=['START','END','IMPEDANCE'])