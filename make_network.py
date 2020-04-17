simplex


class make_nx:

    def __init__(results_dir):
        all_data_dirs = {

        }

    def make_oil(self):
        self.G = nx.DiGraph()
 

        recipe = [
        {'name':'wellpads-pipelines',       'reverse':False,    'dup_1':False, 'dup_2':False,   'desc':'add wellpads -> pipelines'}
        {'name':'oilfields-pipelines',      'reverse':False,    'dup_1':False, 'dup_2':False,   'desc':'add fields -> pipelines'}
        {'name':'pipelines-pipelines',      'reverse':False,    'dup_1':False, 'dup_2':False,   'desc':'add pipelines -> pipelines'}
        {'name':'pipelines-pipelines',      'reverse':True,     'dup_1':False, 'dup_2':False,   'desc':'add pipelines <- pipelines'}
        {'name':'pipelines-ports',          'reverse':False,    'dup_1':False, 'dup_2':False,   'desc':'add pipelines -> ports'}
        {'name':'pipelines-ports',          'reverse':True,     'dup_1':False, 'dup_2':False,   'desc':'add pipelines <- ports'}
        {'name':'ports-shipping',           'reverse':False,    'dup_1':False, 'dup_2':False,   'desc':'add ports -> shipping_lanes'}
        {'name':'ports-shipping',           'reverse':True,     'dup_1':False, 'dup_2':False,   'desc':'add ports <- shipping_lanes'}
        {'name':'shipping-shipping',        'reverse':False,    'dup_1':False, 'dup_2':False,   'desc':'add shipping_lanes -> shipping_lanes'}
        {'name':'shipping-shipping',        'reverse':True,     'dup_1':False, 'dup_2':False,   'desc':'add shipping_lanes <- shipping_lanes'}
        {'name':'pipelines-refineries',     'reverse':False,    'dup_1':False, 'dup_2':False,   'desc':'add pipelines -> refineries'}
        {'name':'pipelines-refineries',     'reverse':True,     'dup_1':True,  'dup_2':False,   'desc':'add refineries -> pipelines_2'}
        {'name':'pipelines-ports',          'reverse':False,    'dup_1':True,  'dup_2':True,    'desc':'add pipelines_2 <-> ports_2 '}
        {'name':'pipelines-ports',          'reverse':True,     'dup_1':True,  'dup_2':True,    'desc':'add pipelines_2 <-> ports_2 '}
        {'name':'ports-shipping',           'reverse':False,    'dup_1':True,  'dup_2':True,    'desc':'add ports_2 <-> shipping_lanes_2'}
        {'name':'ports-shipping',           'reverse':True,     'dup_1':True,  'dup_2':True,    'desc':'add ports_2 <-> shipping_lanes_2'}
        {'name':'shipping-shipping',        'reverse':False,    'dup_1':True,  'dup_2':True,    'desc':'add shipping_lanes_2 <-> shipping_lanes_2'}
        {'name':'shipping-shipping',        'reverse':True,     'dup_1':True,  'dup_2':True,    'desc':'add shipping_lanes_2 <-> shipping_lanes_2'}
        {'name':'cities-pipelines',         'reverse':True,     'dup_1':False, 'dup_2':True,    'desc':'add pipelines_2 -> cities'}
        {'name':'powerstn-pipelines',       'reverse':True,     'dup_1':False, 'dup_2':True,    'desc':'add pipelines_2 -> power_stn'}
        ]


        for step in recipe:
            logging.info(f'doing step {step["desc"]}...')
            dup_strs = ['','']
            order = [1,2]
            if step['dup_1']:
                dup_strs[0]='_B'
            if step['dup_2']:
                dup_strs[1]='_B'
            if step['reverse']:
                order.reverse()
                dup_strs.reverse()

            self.G.add_edges_from([(r[order[0]]+dup_strs[0],r[order[1]]+dup_strs[1],{'z':r[4]}) for r in self.dfs[step['name']].to_records()])
