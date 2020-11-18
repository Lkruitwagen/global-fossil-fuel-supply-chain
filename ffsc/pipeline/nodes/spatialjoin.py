import logging, sys, json, time
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from ffsc.pipeline.nodes.mp_sjoin_mindist import *

N_WORKERS=12

def spatialjoin(df_linear, df_buffered):

    linear_asset_name = df_linear.iloc[0]['unique_id'].split('_')[0]
    point_asset_name = df_buffered.iloc[0]['unique_id'].split('_')[0]
    logger = logging.getLogger(f'spatial_join_{linear_asset_name}_{point_asset_name}')
       
    # sjoin to self
    intersection_df = pd.DataFrame(
                            mp_sjoin_mindist(
                                df_linear=df_linear, 
                                df_buffered=df_buffered, 
                                left_geom_column='geometry', 
                                right_geom_column='geometry', 
                                right_buffer_column='buffer_geom',
                                N_workers=N_WORKERS, 
                                logger=logger,
                                include_min_dist=True),
                            columns=['L_idx','R_idx','intersects', 'NEAREST_PT', 'DISTANCE'])  

    intersection_df = intersection_df[intersection_df['intersects']==True]
    
    intersection_df = intersection_df.sort_values("DISTANCE").groupby("L_idx", as_index=False).first()
    
    intersection_df['START'] = df_linear.iloc[intersection_df['R_idx'].values,df_linear.columns.get_loc('unique_id')].values
    intersection_df['END'] = df_buffered.iloc[intersection_df['L_idx'].values,df_buffered.columns.get_loc('unique_id')].values
    
    return intersection_df[['START','END','NEAREST_PT','DISTANCE']]
    
    



