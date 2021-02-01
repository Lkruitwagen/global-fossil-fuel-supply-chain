import yaml
from math import pi

SEACOST = 800/9000/28

def _cap_factor(N):
    return 1/sum([1/(1.1**ii) for ii in range(N)])

params = dict(
            tperTJ = {
                    'gas': 1e3/52,
                    'coal': 1e3/29.3,
                    'oil': 1e3/41.87,
                    },
            SEACOST = SEACOST, #$/t/km
            RAILCOST = SEACOST*0.8,
            ROADCOST = SEACOST*2,
            SEALOAD = 800/28*0.05, #$/t
            RAILLOAD = 800/28*0.1, #$/t
            ROADLOAD = 800/28*0.2, #$/t

            ## oil pipelines: $2mn/km@35years, + $78.46 / MnBBL-mi -> 206304 $/yr for a 30" pipe at 3m/s flow rate + $78.46/MnBBL-mi
            OIL_PIPELINE = 2000000*_cap_factor(35) / (pi*(30*2.54/100/2)**2 * 3 * .9 * 3600 * 8760) + 78.46 / 6.12 / 1e3 / 1.6, #$_fin/km/yr + $_opx/km/yr
            # gas pipeline: same $_fin, + 700BTU/ton-mile @ $4/mmbtu
            GAS_PIPELINE = 2000000*_cap_factor(35) / (pi*(30*2.54/100/2)**2 * 3 * .9 * 3600 * 8760) + 4 / 1e6 * 700 / 1.6 /.907,

            # LNG: $1000/t/yr cap cost + $0.6/mmbtu opex + 15% parasitic losses
            LNG_TRANSFER =  1000*_cap_factor(25) + 0.6 / 1.055 * 52 + 4/1.055*52 *0.15,  # $/t + 4$/mmbtu / 1.055 -> x$/GJ * 52GJ/t
            check_paths = True,
            COMMUNITY_LEVEL={'coal':7,'oil':9,'gas':9},
            constrain_production = False,
)

yaml.dump(params, open('./conf/base/flow_parameters.yml','w'))