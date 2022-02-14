import os
import cv2
from tqdm import tqdm

video_path = '/run/user/1000/gvfs/smb-share:server=ipl-noaa.local,share=homes/rail/Middleton_2021'

videos = os.listdir(video_path)

for video in tqdm(videos):
    if 'mp4' not in video:
        continue
    print('video_name:', video)
    save_dir = os.path.join(video_path, video[:-4])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    vidcap = cv2.VideoCapture(os.path.join(video_path, video))
    success, image = vidcap.read()
    count = 0

    # print('fps: ', vidcap.get(cv2.CAP_PROP_FPS))

    while success:
        cv2.imwrite(os.path.join(save_dir, "frame_%d.jpg" % count), image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

