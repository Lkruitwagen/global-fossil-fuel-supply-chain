import logging, os, sys, pickle

import networkx as nx
import pandas as pd
from math import pi
import numpy as np

from .recipes import recipes
from .simplex import network_simplex

logger=logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class make_nx:

    def __init__(self, carrier, recipes_used = None):
        
        
        ############################################
        ########## Changes made by Aaron ###########
        ############################################
        
        ### optional third argument of recipes used is added 
        
        # by default, set to the name of the carrier
        
        if recipes_used == None:
            recipes_used = carrier
        ############################################
        
        
        self.all_data_dirs = {
            'cities-N':              os.path.join(os.getcwd(),'results_backup','simplify','cities_nodes_dataframe.csv'),
            'pipelines-cities':      os.path.join(os.getcwd(),'results_backup','simplify','cities_pipelines_edge_dataframe.csv'),
            'ports-cities':          os.path.join(os.getcwd(),'results_backup','output','cities_ports_edge_dataframe.csv'),
            'railways-cities':       os.path.join(os.getcwd(),'results_backup','simplify','cities_railways_edge_dataframe_alt.csv'),
            'coalmines-railways':    os.path.join(os.getcwd(),'results_backup','simplify','coal_mine_railway_edge_dataframe.csv'),
            'coalmines-N':           os.path.join(os.getcwd(),'results_backup','output','coal_mines_nodes_dataframe.csv'),
            'lng-N':                 os.path.join(os.getcwd(),'results_backup','output','lng_nodes_dataframe.csv',  ),
            'lng-pipelines':         os.path.join(os.getcwd(),'results_backup','simplify','lng_pipeline_edge_dataframe.csv'),
            'lng-shipping':          os.path.join(os.getcwd(),'results_backup','output','lng_shipping_route_edge_dataframe.csv'),
            'oilfields-pipelines':   os.path.join(os.getcwd(),'results_backup','simplify','oil_field_edge_dataframe.csv'),
            'oilfields-N':           os.path.join(os.getcwd(),'results_backup','output','oil_field_nodes_dataframe.csv'),
            'pipelines-pipelines':   os.path.join(os.getcwd(),'results_backup','simplify','pipeline_edge_dataframe.csv'),
            'pipelines-N':           os.path.join(os.getcwd(),'results_backup','simplify','pipeline_node_dataframe.csv'),
            'ports-N':               os.path.join(os.getcwd(),'results_backup','output','port_node_dataframe.csv',  ),
            'ports-pipelines':       os.path.join(os.getcwd(),'results_backup','simplify','port_pipeline_edge_dataframe.csv'),
            'ports-shipping':        os.path.join(os.getcwd(),'results_backup','output','port_ship_edge_dataframe.csv'),
            'ports-railways':        os.path.join(os.getcwd(),'results_backup','simplify','port_railway_edge_dataframe.csv'),
            'powerstn-N':            os.path.join(os.getcwd(),'results_backup','output','power_station_nodes_dataframe.csv'),
            'powerstn-pipelines':    os.path.join(os.getcwd(),'results_backup','simplify','power_station_pipeline_edge_dataframe.csv'),
            'powerstn-railways':     os.path.join(os.getcwd(),'results_backup','simplify','power_station_railway_edge_dataframe.csv'),
            'procplant-N':           os.path.join(os.getcwd(),'results_backup','output','processing_plant_nodes_dataframe.csv'),
            'procplant-pipelines':   os.path.join(os.getcwd(),'results_backup','simplify','processing_plant_pipeline_edge_dataframe.csv'),
            'railways-railways':     os.path.join(os.getcwd(),'results_backup','simplify','railway_edge_dataframe.csv'),
            'railways-N':            os.path.join(os.getcwd(),'results_backup','simplify','railway_nodes_dataframe.csv'),
            'refineries-N':          os.path.join(os.getcwd(),'results_backup','output','refinery_nodes_dataframe.csv'),
            'refineries-pipelines':  os.path.join(os.getcwd(),'results_backup','simplify','refinery_pipeline_edge_dataframe.csv'),
            'shipping-shipping':     os.path.join(os.getcwd(),'results_backup','output','shipping_edge_dataframe.csv'),
            'shipping-N':            os.path.join(os.getcwd(),'results_backup','output','shipping_node_dataframe.csv'),
            'wellpads-N':            os.path.join(os.getcwd(),'results_backup','output','well_pad_nodes_dataframe.csv'),
            'wellpads-pipelines':    os.path.join(os.getcwd(),'results_backup','simplify','well_pad_pipeline_edge_dataframe.csv'),
        }

        self.G = nx.DiGraph()

        self.carrier = carrier

        self.recipes = recipes
        
        self.recipes_used = recipes_used ##


        tperTJ = {
        'gas': 1e3/52,
        'coal': 1e3/29.3,
        'oil': 1e3/41.87,
        }

        SEACOST = 800/9000/28 #$/t/km
        RAILCOST = SEACOST*0.8
        SEALOAD = 800/28*0.05 #$/t
        RAILOAD = 800/28*0.1 #$/t

        ## oil pipelines: $2mn/km@35years, + $78.46 / MnBBL-mi -> 206304 $/yr for a 30" pipe at 3m/s flow rate + $78.46/MnBBL-mi
        OIL_PIPELINE = 2000000*self._cap_factor(35) / (pi*(30*2.54/100/2)**2 * 3 * .9 * 3600 * 8760) + 78.46 / 6.12 / 1e3 / 1.6 #$_fin/km/yr + $_opx/km/yr
        # gas pipeline: same $_fin, + 700BTU/ton-mile @ $4/mmbtu
        GAS_PIPELINE = 2000000*self._cap_factor(35) / (pi*(30*2.54/100/2)**2 * 3 * .9 * 3600 * 8760) + 4 / 1e6 * 700 / 1.6 /.907

        # LNG: $1000/t/yr cap cost + $0.6/mmbtu opex + 15% parasitic losses
        LNG_TRANSFER =  1000*self._cap_factor(25) + 0.6 / 1.055 * 52 + 4/1.055*52 *0.15  # $/t + 4$/mmbtu / 1.055 -> x$/GJ * 52GJ/t





        self.impedance_factors = {        
                            'pipelines-cities':     {'gas':0,'coal':0,'oil':0},
                            'ports-cities':         {'gas':0,'coal':0,'oil':0},
                            'railways-cities':      {'gas':0,'coal':0,'oil':0},
                            'coalmines-railways':   {'gas':0,'coal':0,'oil':0},
                            'lng-pipelines':        {'gas':0,'coal':0,'oil':0},
                            'lng-shipping':         {'gas':LNG_TRANSFER*tperTJ['gas'],'coal':0,'oil':0},
                            'oilfields-pipelines':  {'gas':0,'coal':0,'oil':0},
                            'pipelines-pipelines':  {'gas':GAS_PIPELINE*tperTJ['gas'],'coal':0,'oil':OIL_PIPELINE*tperTJ['oil']},
                            'ports-pipelines':      {'gas':0,'coal':0,'oil':0},
                            'ports-shipping':       {'gas':SEALOAD*tperTJ['gas'],'coal':SEALOAD*tperTJ['coal'],'oil':SEALOAD*tperTJ['oil']},
                            'ports-railways':       {'gas':0,'coal':0,'oil':0},
                            'powerstn-pipelines':   {'gas':0,'coal':0,'oil':0},
                            'powerstn-railways':    {'gas':0,'coal':0,'oil':0},
                            'procplant-pipelines':  {'gas':0,'coal':0,'oil':0},
                            'railways-railways':    {'gas':0,'coal':RAILCOST*tperTJ['coal'],'oil':0},
                            'refineries-pipelines': {'gas':0,'coal':0,'oil':0},
                            'shipping-shipping':    {'gas':SEACOST*tperTJ['gas'],'coal':SEACOST*tperTJ['coal'],'oil':SEACOST*tperTJ['oil']},
                            'wellpads-pipelines':   {'gas':0,'coal':0,'oil':0},
                            }

        
        ############################################
        ########## Changes made by Aaron ###########
        ############################################
        
        ### print (self.impedance_factors) ### Aaron has commented this out
        
        # raise value error unless both valid carrier and valid recipes are used
        # this ensures that the _load_dfs() will operate correctly
        
        valid_carrier = ['oil', 'gas', 'coal']
        
        valid_recipes = []
        for key in recipes:
                valid_recipes.append(key)
        
        if self.carrier in valid_carrier:
            print('carrier: ' + self.carrier)
        else:
            raise ValueError("make_nx: carrier must be one of %r." % valid_carrier)
            
        if self.recipes_used in valid_recipes:
            print('recipes used: ' + self.recipes_used)
        else:
            raise ValueError("make_nx: recipes must be one of %r." % valid_recipes)
        ############################################
        


    @staticmethod
    def _cap_factor(N):
        return 1/sum([1/(1.1**ii) for ii in range(N)])

    def _load_dfs(self):
        self.dfs = {}

        self.dfs['cities'] = pd.read_csv(self.all_data_dirs['cities-N'])
        self.dfs['powerstns'] = pd.read_csv(self.all_data_dirs['powerstn-N'])

        keys = [kk['name'] for kk in self.recipes[self.recipes_used]] ##

        for kk in keys:
            self.dfs[kk] = pd.read_csv(self.all_data_dirs[kk])


    def _fill_graph(self):

        for step in self.recipes[self.recipes_used]: ##
            logger.info(f'doing step {step["desc"]}...')
            dup_strs = ['','']
            start_col = [cc for cc in self.dfs[step['name']].columns if 'START_ID' in cc][0]
            end_col = [cc for cc in self.dfs[step['name']].columns if 'END_ID' in cc][0]
            order = [start_col,end_col]
            if step['dup_1']:
                dup_strs[0]='_B'
            if step['dup_2']:
                dup_strs[1]='_B'
            if step['reverse']:
                order.reverse()
                dup_strs.reverse()


            
            #if 'impedance' not in self.dfs[step['name']].columns:
            #    print ('missing impedance')
            #    print(self.dfs[step['name']].columns.tolist())
            #    self.dfs[step['name']]['impedance'] = 1

            if 'distance' not in self.dfs[step['name']].columns:
                logger.info(f'{step["name"]} missing distance')
                logger.info(self.dfs[step['name']].columns.tolist())
                self.dfs[step['name']]['distance'] = 1


            #print (self.dfs[step['name']].loc[:,order+['distance']].values)

            self.G.add_edges_from(
                [
                    ((r[0]+dup_strs[0]).strip(),(r[1]+dup_strs[1]).strip(),{'z':int(round(r[2]*self.impedance_factors[step['name']][self.carrier]))})  ##
                for r in self.dfs[step['name']].loc[:,order+['distance']].values.tolist()]
                )

            #logger.info(f'r in G {"r" in self.G.nodes}')

        all_impedances = [e[2]['z'] for e in self.G.edges(data=True)]

        print (nx.info(self.G))
        print ('impedance: mean',np.mean(all_impedances),'max',np.max(all_impedances),'min',np.min(all_impedances))

    def _prep_flow(self):

        # add supersource
        if self.carrier=='coal':
            source_nodes = [n for n in self.G.nodes if 'coal_mine' in n]
        else:

            # Fix Fix fix
            source_nodes = [n for n in self.G.nodes if ('oil_field' in n) or ('well_pad' in n)]

        self.G.add_node('supersource')

        #print ('source nodes')#
        #print (source_nodes)

        for source_node in source_nodes:
            self.G.add_edge('supersource',source_node,z=0)


        ### add demand
        nx.set_node_attributes(self.G, 0, 'D')

        self.dfs['cities']['demand'] = self.dfs['cities']['total_'+self.carrier+'_consumption'].fillna(0) / 1e3 # TJ/yr


        city_nodes = [n for n in self.G.nodes if 'city' in n]

        # from cities
        scope_cities = [city_id for city_id in self.dfs['cities'].loc[:,['CityNodeId:ID(CityNode)','demand']].values.tolist() if city_id[0] in city_nodes]
        #print ('scope cities', len(scope_cities), len(self.dfs['cities']))

        attrs = {city:{'D':int(round(demand))} for city, demand in scope_cities}

        nx.set_node_attributes(self.G, attrs)

        self.dfs['powerstns']['demand']=0

        if self.carrier=='coal':
            self.dfs['powerstns'].loc[self.dfs['powerstns']['fuel1']=='Coal','demand'] = self.dfs['powerstns'].loc[self.dfs['powerstns']['fuel1']=='Coal','capacity_mw']*8760*.6 * 3.6 / 1e3 # TJ/yr
        elif self.carrier=='gas':
            self.dfs['powerstns'].loc[self.dfs['powerstns']['fuel1']=='Gas','demand'] = self.dfs['powerstns'].loc[self.dfs['powerstns']['fuel1']=='Gas','capacity_mw']*8760*.6 * 3.6 / 1e3 # TJ/yr
        elif self.carrier=='oil':
            self.dfs['powerstns'].loc[self.dfs['powerstns']['fuel1']=='Oil','demand'] = self.dfs['powerstns'].loc[self.dfs['powerstns']['fuel1']=='Oil','capacity_mw']*8760*.6 * 3.6 / 1e3 # TJ/yr

        pwrstn_nodes = [n for n in self.G.nodes if 'power_station' in n]

        # from powerstns
        scope_powerstns = [pwrstn for pwrstn in self.dfs['powerstns'].loc[:,['PowerStationID:ID(PowerStation)','demand']].values.tolist() if pwrstn[0] in pwrstn_nodes]
        print ('scope power stations)', len(scope_powerstns), len(self.dfs['powerstns']))

        attrs = {pwrstn:{'D':int(round(demand))} for pwrstn, demand in scope_powerstns}

        nx.set_node_attributes(self.G, attrs)

        

        logger.info(f'checking powerstation paths...')

        p_count = 0
        c_count=0

        pathless_pwrstns = []
        for ii_p, (pwrstn, demand) in enumerate(scope_powerstns):
            if ii_p %1000==0:
                logger.info(f'ii_p {ii_p}, p_count {p_count}')
            if not nx.has_path(self.G, 'supersource',pwrstn):
                #logger.info(f'No Path! {pwrstn}')
                p_count +=1
                pathless_pwrstns.append(pwrstn)

        logger.info(f'checking city paths...')
        pathless_cities = []
        for ii_c, (city, demand) in enumerate(scope_cities):
            if ii_c %1000==0:
                logger.info(f'ii_c {ii_c}, c_count {c_count}')
            if not nx.has_path(self.G, 'supersource',city):
                #logger.info(f'No Path! {city}')
                c_count+=1
                pathless_cities.append(city)


        self.G.remove_nodes_from(pathless_pwrstns)
        self.G.remove_nodes_from(pathless_cities)

        D_cities = sum([demand for city, demand in scope_cities if city not in pathless_cities])
        D_pwrstns = sum([demand for stn, demand in scope_powerstns if stn not in pathless_pwrstns])

        logger.info(f'sum D pwnstns: {D_pwrstns}, sum D cities: {D_cities}')
        logger.info(f'combined sum {D_pwrstns+D_cities}')


        attrs = {'supersource':{'D':-1*sum([self.G.nodes[u].get('D', 0) for u in list(self.G)])}}
        nx.set_node_attributes(self.G, attrs)




    def _solve_flow(self):

        flow_cost, flow_dict = network_simplex(self.G, demand='D', capacity='capacity', weight='z')

        print ('flow cost',flow_cost)
        #print (flow_dict)

        pickle.dump(flow_cost, open('./'+self.carrier+'_flow_cost.pkl','wb'))
        pickle.dump(flow_dict, open('./'+self.carrier+'_flow_dict.pkl','wb'))

    

    

if __name__=="__main__":

    if sys.argv[1] in ['oil','coal','gas']:

        gen = make_nx(sys.argv[1])
        gen._load_dfs()
        gen._fill_graph()
        gen._prep_flow()
        gen._solve_flow()
    else:
        print ('give one of oil, coal or gas')