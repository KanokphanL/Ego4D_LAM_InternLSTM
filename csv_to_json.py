import json
import pandas as pd
import scipy.io as sio
import os
import argparse

def csv2json(args): 
        
    records=[]
    df = pd.read_csv(args.input,header=None)

    d_records = df.to_dict('records')
    dic={"version": "1.0","challenge": "ego4d_looking_at_me"}
    print(dic)
    dic['results']=[]
    leng = len(d_records)

    for i, meta in enumerate(d_records):
        uid = meta[ 0 ]
        trackid = meta[ 2 ]
        unique_id = meta[ 1 ]
        score = meta[ 4 ]
        label = 1
        new_dic = {'video_id': uid, 'unique_id': unique_id, 'track_id': trackid, 'label': label, 'score': score}
        dic[ 'results' ].append(new_dic)
        if i % 1000 == 0:
            print(i, leng)
    print(len(dic['results']))

    with open(args.output, 'w') as f:
        json.dump(dic, f)


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Ego4d Social Benchmark')
    argparser.add_argument('--input', type=str, help='Input pred.csv file')
    argparser.add_argument('--output', type=str, help='Output .json file')
    args = argparser.parse_args()
    csv2json(args)
