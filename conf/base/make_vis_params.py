import yaml
from math import pi



params = dict(
        vis_colors = dict(
            SHIPPINGROUTE=[127, 86, 54],
            PIPELINE=[0,100,0],
            RAILWAY=[67,67,67],
            REFINERY=[190, 70, 166],
            OILFIELD=[0,28,73],
            OILWELL=[108, 0, 147],
            COALMINE=[137, 22, 62],
            LNGTERMINAL=[70, 190, 177],
            PORT=[235, 138, 28],
            POWERSTATION=[138, 194, 126],
            CITY=[150, 195, 228],    
            FINALMILE=[150, 195, 228],  
            ne=[219,219,219],
            MISSING_CITY=[81, 112, 135],
            MISSING_POWERSTATION=[255,0,0],
        ),
        type_style = dict(
            ne           = {'alpha':1., 'zorder':0},
            lin_asset    = {'alpha':0.3, 'linewidth':1,'zorder':1},
            pt_asset     = {'alpha':0.6, 'markersize':5,'zorder':2},
            edges        = {'alpha':0.3, 'linewidth':1,'zorder':1},
            missing      = {'alpha':0.7, 'markersize':6,'zorder':3},
            missing_city = {'alpha':1, 'markersize':6,'zorder':3,'marker':6},
            missing_powerstation = {'alpha':1, 'markersize':6,'zorder':3,'marker':7},
        ),
        community_levels = dict(
            coal=8,
            oil=10,
            gas=11,
        ),
        vis_N_communities=5,
)

yaml.dump(params, open('./conf/base/vis_parameters.yml','w'))