import os
import cv2
from tqdm import tqdm
from IPython import embed

video_path = '/run/user/1000/gvfs/smb-share:server=ipl-noaa.local,share=homes/rail/Middleton_2021'

videos = os.listdir(video_path)


frame_num={}
for video in tqdm(videos):

    if 'mp4' in video:
        continue
    print('video_name:', video)
    frame_num[video] = len(os.listdir(os.path.join(video_path,video)))

embed()


