import subprocess, os, json
import pandas as pd

os.makedirs('data/tmp', exist_ok=True)
os.makedirs('data/videos', exist_ok=True)

# with open('data/json/av_train.json', 'r') as f:
    # ori_annot_train = json.load(f)

with open('data/json/av_test_unannotated.json', 'r') as f:
    ori_annot_val = json.load(f)

with open('data/csv/manifest.csv', 'r') as f:
    manifest = pd.read_csv(f)

for video in ori_annot_val['videos']:
    video_uid = video['video_uid']
    print(f'downloading {video_uid}...')
    canonical_path = manifest[manifest['video_uid']==video_uid]['canonical_s3_location'].iloc[0]
    cmd = f'aws s3 cp {canonical_path} data/tmp/'
    while(not os.path.exists(f'data/tmp/{video_uid}')):
        subprocess.call(cmd, shell=True)

    for clip in video['clips']:
        clip_uid = clip['clip_uid']
        print(f'chunking {clip_uid}...')
        start_sec = clip['video_start_sec']
        end_sec = clip['video_end_sec']
        cmd = f'ffmpeg -y -i data/tmp/{video_uid} -ss {start_sec} -to {end_sec} -c:a copy -vcodec libx264 -keyint_min 2 -g 1  -y data/videos_test/{clip_uid}.mp4'
        subprocess.call(cmd, shell=True)
    # os.remove(f'data/tmp/{video_uid}')

# for video in ori_annot_train['videos']:
#     video_uid = video['video_uid']
#     print(f'downloading {video_uid}...')
#     canonical_path = manifest[manifest['video_uid']==video_uid]['canonical_s3_location'].iloc[0]
#     cmd = f'aws s3 cp {canonical_path} data/tmp/'
#     while(not os.path.exists(f'data/tmp/{video_uid}')):
#         subprocess.call(cmd, shell=True)

#     for clip in video['clips']:
#         clip_uid = clip['clip_uid']
#         print(f'chunking {clip_uid}...')
#         start_sec = clip['video_start_sec']
#         end_sec = clip['video_end_sec']
#         cmd = f'ffmpeg -y -i data/tmp/{video_uid} -ss {start_sec} -to {end_sec} -c:a copy -vcodec libx264 -keyint_min 2 -g 1  -y data/videos/{clip_uid}.mp4'
#         subprocess.call(cmd, shell=True)
#     # os.remove(f'data/tmp/{video_uid}')

print('finish')
