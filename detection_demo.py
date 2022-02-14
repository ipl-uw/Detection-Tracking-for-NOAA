import csv
from IPython import embed
import cv2
import os
import numpy as np
from tqdm import tqdm

### 0. set frames segment to show###
fps = 15   #视频帧率
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
color = (0, 255,0)
thickness=10

### 1. read detection csv file ###
file_path = '/home/jiemei/Documents/rail_detection/detection_result/20180602T152105_master/detection_result_with_blank_frames-sample.csv'
frame_path = '/run/user/1000/gvfs/smb-share:server=ipl-noaa.local,share=home/aukebay2018_sample_data/20180602T152105_master'
save_dir = '/home/jiemei/Documents/rail_detection/detection_result/20180602T152105_master/demo'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

file = csv.reader(open(file_path,'r'))
raw0 = []
for i, line in enumerate(file):
    if i==0: # skip first row, which is header
        continue
    else:
        raw0.append(line)

raw = np.array(raw0)

R = raw[:, 0].copy()
all_frame_name = np.unique(R)

img = cv2.imread(os.path.join(frame_path, all_frame_name[0]))

videoWriter = cv2.VideoWriter(save_dir+'/'+file_path.split('/')[-1].replace('.csv', '_demo.avi'), fourcc, fps, (img.shape[1], img.shape[0]))

for frame_name in tqdm(all_frame_name):
    idxs = raw[:, 0] == frame_name
    img = cv2.imread(os.path.join(frame_path, frame_name))

    rows = raw[idxs]

    for row in rows:
        if row[5]=='0': ### no detections in this frame
            continue
        cv2.rectangle(img, (int(float(row[1])), int(float(row[2]))), (int(float(row[3])), int(float(row[4]))), color, thickness)
        cv2.putText(img, row[-1], (int(float(row[1])), int(float(row[4])) + 30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        cv2.putText(img, row[5][:4], (int(float(row[1])) + 80, int(float(row[4])) + 30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

    videoWriter.write(img)
videoWriter.release()





# with open(file_path) as csvfile:
#     csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
#     birth_header = next(csv_reader)  # 读取第一行每一列的标题
#     for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
#
#         if int(row[0].split('_')[0])== int(start_frame.split('_')[0]):
#             img = cv2.imread(os.path.join(frame_path, start_frame))
#             videoWriter = cv2.VideoWriter(file_path.replace('.csv', '_demo'+start_frame.split('_')[0]+'.avi'), fourcc, fps, (img.shape[1], img.shape[0]))
#
#         if int(row[0].split('_')[0]) >= int(start_frame.split('_')[0]):
#             img = cv2.imread(os.path.join(frame_path, row[0]))
#
#             # if float(row[5])>=0.91:
#
#             cv2.rectangle(img, (int(float(row[1])), int(float(row[2]))),(int(float(row[3])), int(float(row[4]))), color, thickness)
#             cv2.putText(img, row[-1], (int(float(row[1])), int(float(row[4])) + 30),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
#             cv2.putText(img, row[5][:4], (int(float(row[1]))+80, int(float(row[4])) + 30),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
#
#             videoWriter.write(img)
#
#         if row[0] ==end_frame:
#             break
#
# videoWriter.release()




