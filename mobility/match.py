import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
import json
from census_tract_locator import *


def get_chicago_poi(path):
    raw = pd.read_csv(path)
    chicago_poi = raw[raw['city'] == 'Chicago']
    return chicago_poi


def get_chi_pattern_csv(PATTERN_ROOT):
    path_list = []
    for year in [2018, 2019]:
        for month in range(1, 13):
            path_list.append(
                PATTERN_ROOT+str(year)+'_'+str(month)+'.csv')
    return path_list


def get_tract_dic(PATTERN_PATH):
    locator = CensusTract(
        tract_data_dir='tract_boundaries_us_for_locator.pickle')
    chi_pattern_csv = get_chi_pattern_csv()
    chi_poi = pd.read_csv(chi_poi_path)

    # get columns
    test = pd.read_csv(PATTERN_PATH, chunksize=1)
    for index, chunk in enumerate(test):
        df = chunk
        break
    col = ['index'] + list(df.columns)

    from collections import defaultdict
    chi_pattern_dic = defaultdict(int)

    for path in tqdm(chi_pattern_csv):
        pattern = pd.read_csv(path)
        pattern.columns = col

        for i in tqdm(range(pattern.shape[0])):
            poi_id = pattern['safegraph_place_id'][i]
            poi_query = chi_poi[chi_poi['safegraph_place_id'] == poi_id]
            if poi_query.shape[0] != 0:
                poi_query = poi_query.reset_index(drop=True)
                lon = poi_query['longitude'][0]
                lat = poi_query['latitude'][0]
                poi_tract = locator.get_tract(lat=lat, lon=lon)
                visitor_home_dic = json.loads(pattern['visitor_home_cbgs'][i])
                if visitor_home_dic != {}:
                    for item in visitor_home_dic:
                        chi_pattern_dic[(poi_tract, item)
                                        ] += visitor_home_dic[item]
        with open(dic_path, 'wb') as f:
            pickle.dump(chi_pattern_dic, f)


def get_tract_to_tract_dic(FIP_PATH):
    with open(dic_path, 'rb') as f:
        chi_dic = pickle.load(f)
    with open(FIP_PATH, 'rb') as f:
        u2v_fips = pickle.load(f)
    from collections import defaultdict
    chi_dic_ct = defaultdict(int)
    for item in tqdm(chi_dic):
        if ((int(item[0]) in u2v_fips) and (int(item[1][:11]) in u2v_fips)):
            chi_dic_ct[(int(item[0]), int(item[1][:11]))] += chi_dic[item]
    with open(ct_dic_path, 'wb') as f:
        pickle.dump(chi_dic_ct, f)


def get_postive_pair(taxi, chi_fips_idx):
    # taxi = taxi[YEAR]
    pair = []
    anchor_tract_list = []
    for key in tqdm(taxi.keys()):
        try:
            anchor = chi_fips_idx[key[0]]
            pos = chi_fips_idx[key[1]]
            if anchor not in anchor_tract_list:
                anchor_tract_list.append(anchor)
            if (taxi[key] >= 100):
                # trips distribute sparsely between census tracts, e.g. 1 trip to 10k trips
                # use sqrt to reduce the number of trips a bit, could try log and other methods
                count = int(np.log(taxi[key]))
                for i in range(count):
                    pair.append([pos, anchor])
        except:
            continue
    return pair, anchor_tract_list


def get_context(chi_fips_idx, pair):
    context = {}
    for i in range(len(chi_fips_idx)):
        context[i] = []
    for item in tqdm(pair):
        pos = item[0]
        anchor = item[1]
        context[anchor].append(pos)
    return context


def get_count(chi_fips_idx, pair):
    count = np.zeros(len(chi_fips_idx))
    for item in tqdm(pair):
        count[item[0]] += 1
    return count

