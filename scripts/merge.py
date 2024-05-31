import os
import json


def merge_json(path_results, path_merges):
    d = {}
    with open(path_merges, "w+", encoding="utf-8") as f0:
        
        with open(os.path.join(path_results, 'av_train.json'), "r", encoding="utf-8") as f1:
            json_dict = json.load(f1)
            d = json_dict.copy()
        with open(os.path.join(path_results, 'av_val.json'), "r", encoding="utf-8") as f2:
            json_dict = json.load(f2)
            vs = json_dict['videos']
            for v in vs:
                d['videos'].append(v)
        json.dump(d, f0)
