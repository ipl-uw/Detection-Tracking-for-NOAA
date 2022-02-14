import csv
from IPython import embed
import cv2
import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from huber_regression import huber_regression
from util import save_to_csv


def load_mot_all_frames(detections, all_frame_names):
    '''
    :param detections: tracking.csv
    :param all_frame_names:  all frames including frames without detections/not in the tracking.csv
    :return: data = [[{},{},..], [], []...] one list represents one frame, one dict represents one detections in this frame, data has all frames inside, if there is no
    detection in one frame, then [{'id': '', 'bbox': ('', '','','',), 'score': '', 'class': '','frame_name': frame_name}]
    '''
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
    'w', 'h', 'score']).

    Args:
        detections (str, numpy.ndarray): path to csv file containing the detections or numpy array containing them.
        nms_overlap_thresh (float, optional): perform non-maximum suppression on the input detections with this thrshold.
                                              no nms is performed if this parameter is not specified.
        with_classes (bool, optional): indicates if the detections have classes or not. set to false for motchallange.
        nms_per_class (bool, optional): perform non-maximum suppression for each class separately

    Returns:
        list: list containing the detections for each frame.
    """
    # if nms_overlap_thresh:
    #     assert with_classes, "currently only works with classes available"

    data = []

    # original code to get raw
    # if type(detections) is str:
    #     raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
    #     if np.isnan(raw).all():
    #         raw = np.genfromtxt(detections, delimiter=' ', dtype=np.float32)
    #
    # else:
    #     # assume it is an array
    #     assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
    #     raw = detections.astype(np.float32)

    # Jie Mei code to get raw and data

    file = csv.reader(open(detections, 'r'))
    raw0 = []
    for i, line in enumerate(file):
        if i == 0:  # skip first row, which is header
            continue
        else:
            raw0.append(line)

    raw = np.array(raw0)
    if len(raw)==0:
        return data

    # delete redundant name
    # R = raw[:, 1].copy()
    # all_frame_name = np.unique(R)

    print('loading csv trackings...')
    for frame_name in tqdm(all_frame_names):



        idx = raw[:, 1] == frame_name
        if True not in idx:  ### means this frame is not saved in tracking.csv
            dets = [{'id': '', 'bbox': ('', '', '', '',), 'score': '', 'class': '', 'frame_name': frame_name}]
            data.append(dets)
            continue

        scores = raw[idx, 6].astype(float)
        bbox = raw[idx, 2:6].astype(float)
        classes = ['fish'] * bbox.shape[0]
        track_ids = raw[idx, 0].astype(float)

        dets = []
        for bb, s, c, id in zip(bbox, scores, classes, track_ids):
            dets.append(
                {'id': id, 'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, 'class': c, 'frame_name': frame_name})
        data.append(dets)
    print('loading csv tracking... Done!')

    return data


def save_as_one_video(image_path, save_dir, data, fps, box_thickness, txt_thickness, fontScale):
    print('Generating tracking video...')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    path = os.listdir(image_path)
    img_example = cv2.imread(image_path + '/' + path[0])
    size = img_example.shape[:2][::-1]
    videoWriter = cv2.VideoWriter(save_dir + '/tracking_demo_all_tracks.avi', fourcc, fps, size)  # (1360,480)为视频大小
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    for detections_frame in tqdm(data):
        frame_path = image_path + '/' + detections_frame[0]['frame_name']  # only read frames with detections
        img = cv2.imread(frame_path)

        if detections_frame[0]['id'] == '':
            videoWriter.write(img)
            continue

        for det in detections_frame:
            c = int(det['id']) % 3
            color = colors[c]

            xmin = int(float(det['bbox'][0]))
            ymin = int(float(det['bbox'][1]))
            xmax = int(float(det['bbox'][2]))
            ymax = int(float(det['bbox'][3]))

            # print(det)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, box_thickness)
            cv2.putText(img, 'fish', (xmin, ymax + int((ymax - ymin) * 0.2)), cv2.FONT_HERSHEY_COMPLEX, fontScale,
                        color, txt_thickness)
            cv2.putText(img, str(np.round(det['score'], 3)),
                        (xmax - int((xmax - xmin) * 0.2), ymax + int((ymax - ymin) * 0.2)), cv2.FONT_HERSHEY_COMPLEX,
                        fontScale, color, txt_thickness)
            cv2.putText(img, 'id: ' + str(int(det['id'])), (xmin, ymin - int((ymax - ymin) * 0.2)),
                        cv2.FONT_HERSHEY_COMPLEX, fontScale, color, txt_thickness)
            # 写入视频
        videoWriter.write(img)

    videoWriter.release()
    print('saving done!')


# save as mutiple tracks videos
def read_track_dict(file_name, col_name):
    csvFile_all = open(file_name, 'r')
    track_csv = csv.DictReader(csvFile_all)

    track_dict = OrderedDict()
    for i, row in enumerate(track_csv):
        # embed()
        track_id = row[col_name]

        if track_id not in track_dict:
            track_dict[track_id] = []
        track_dict[track_id].append(row)

    csvFile_all.close()

    return track_dict

import shutil
def save_as_videos(image_path, save_dir, track_dict, fps, box_thickness, txt_thickness, fontScale):
    print('saving tracks as video...')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    path = os.listdir(image_path)
    img_example = cv2.imread(image_path + '/' + path[0])
    size = img_example.shape[:2][::-1]

    for j,track_id in tqdm(enumerate(track_dict)):
        if j>=5:
            break
        each_track = track_dict[track_id]  # alist

        videoWriter = cv2.VideoWriter(save_dir + '/' + track_id + '.avi', fourcc, fps, size)  # (1360,480)为视频大小

        for each_frame in each_track:
            # print('ok')
            # 读取当前track的当前 frame            os.mkdir(save_dir)，进行画图
            img = cv2.imread(image_path + '/' + each_frame['filename'])
            color = (0, 255, 0)

            txt = 'track id:' + track_id
            xmin = int(float(each_frame['xmin']))
            ymin = int(float(each_frame['ymin']))
            xmax = int(float(each_frame['xmax']))
            ymax = int(float(each_frame['ymax']))

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, box_thickness)
            cv2.putText(img, each_frame['class'], (xmin, ymax + int((ymax - ymin) * 0.2)), cv2.FONT_HERSHEY_COMPLEX,
                        fontScale, color, txt_thickness)
            cv2.putText(img, each_frame['conf'][:4], (xmax - int((xmax - xmin) * 0.2), ymax + int((ymax - ymin) * 0.2)),
                        cv2.FONT_HERSHEY_COMPLEX, fontScale, color, txt_thickness)
            cv2.putText(img, txt, (int(size[0] / 8), int(size[1] / 8)), cv2.FONT_HERSHEY_COMPLEX, fontScale, color,
                        txt_thickness)
            # 写入视频
            videoWriter.write(img)

            # 多停留几秒
            # if each_frame['max_bbox'] == 1:
            #     for i in range(5):
            #         videoWriter.write(img)

        videoWriter.release()

    print('saving done!')


def kept_or_not(last_frame, kept_ROI):
    '''

    Args:
        last_frame:
        kept_ROI: (xmin, ymin, xmax, ymax)

    Returns:

    '''
    xmin = kept_ROI[0]
    ymin = kept_ROI[1]
    xmax = kept_ROI[2]
    ymax = kept_ROI[3]

    xc = (last_frame['bbox'][0] + last_frame['bbox'][2]) / 2
    yc = (last_frame['bbox'][1] + last_frame['bbox'][3]) / 2

    if xmin <= xc <= xmax and ymin <= yc <= ymax:
        return 1
    else:
        return -1

    # first_frame_bbox_xc = (float(track[0]['bbox'][0]) + float(track[0]['bbox'][2])) / 2
    # first_frame_bbox_yc = (float(track[0]['bbox'][1]) + float(track[0]['bbox'][3])) / 2
    # last_frame_bbox_xc = (float(track[-1]['bbox'][0]) + float(track[-1]['bbox'][2])) / 2
    # last_frame_bbox_yc = (float(track[-1]['bbox'][1]) + float(track[-1]['bbox'][3])) / 2
    #
    # if direction == 'l2r':
    #     if first_frame_bbox_xc > last_frame_bbox_xc:
    #         flag = 1
    #     else:
    #         flag = 0
    # elif direction == 'r2l':
    #     if first_frame_bbox_xc < last_frame_bbox_xc:
    #         flag = 1
    #     else:
    #         flag = 0
    # elif direction == 'b2t':
    #     if first_frame_bbox_yc > last_frame_bbox_yc:
    #         flag = 1
    #     else:
    #         flag = 0
    # elif direction == 't2b':
    #     if first_frame_bbox_yc < last_frame_bbox_yc:
    #         flag = 1
    #     else:
    #         flag = 0

    # return flag


def calculate_kept_ROI(ROI, direction, coef=0.25):
    '''

    Args:
        ROI: (x,y,w,h), xy are top left corner
        direction: 'l2r', 'r2l', 'b2t', 't2b'

    Returns:Kept_ROI (xmin, ymin, xmax, ymax)

    '''

    xmin = ROI[0]
    ymin = ROI[1]
    xmax = ROI[0] + ROI[2]
    ymax = ROI[1] + ROI[3]

    if direction == 'l2r':
        Kept_ROI = (xmin + (xmax - xmin) * (1 - coef), ymin, xmax, ymax)

    elif direction == 'r2l':
        Kept_ROI = (xmin, ymin, xmin + (xmax - xmin) * coef, ymax)

    elif direction == 'b2t':
        Kept_ROI = (xmin, ymin, xmax, ymin + (ymax - ymin) * coef)

    elif direction == 't2b':
        Kept_ROI = (xmin, ymin + (ymax - ymin) * (1 - coef), xmin, ymax)

    return Kept_ROI


def save_data_huber_to_csv(save_dir, data_huber, direction, ROI):
    '''

    Args:
        save_dir: csv save dir
        data_huber: [[{},{},..], [], []...] one list represents one frame, one dict represents one detections in this frame, data has all frames inside, if there is no
    detection in one frame, then [{'id': '', 'bbox': ('', '','','',), 'score': '', 'class': '','frame_name': frame_name}]
        ROI: detection ROI, (x,y,w,h), will be used to calculate kept_ROI

    Returns:

    '''
    print('saving huber regreesion result to csv...')

    kept_ROI = calculate_kept_ROI(ROI, direction, coef=0.4)

    # organize data_huber into: { 'track_id': [{},{},{}], :[],...}
    tracks = {}

    # get all existing track id, which may be not continuous because huber merge some tracks
    existing_id = []
    for frame in data_huber:

        for det_dict in frame:
            if det_dict['id'] not in existing_id and det_dict['id'] != '':  # skip no detections frames
                existing_id.append(det_dict['id'])

    # find detections with the same track id and put all of them into one track
    new_track_id = 1
    for exist_id in existing_id:
        tracks[new_track_id] = []
        for frame in data_huber:
            for det_dict in frame:
                if det_dict['id'] == exist_id:
                    tracks[new_track_id].append(det_dict)
        new_track_id += 1

    # save each track into csv, e.g. track 0: frame 012345, track 1: frame 345678
    import csv

    with open(save_dir, "w") as csvfile:
        field_names = ['id', 'filename', 'xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class', 'length', 'kept']
        odict = csv.DictWriter(csvfile, field_names)
        odict.writeheader()

        for track_id in tracks.keys():
            track = tracks[track_id]

            kept = kept_or_not(track[-1], kept_ROI)

            for frame_det in track:
                row = {'id': int(track_id),
                       'filename': frame_det['frame_name'],
                       'xmin': int(frame_det['bbox'][0]),
                       'ymin': int(frame_det['bbox'][1]),
                       'xmax': int(frame_det['bbox'][2]),
                       'ymax': int(frame_det['bbox'][3]),
                       'conf': frame_det['score'],
                       'class': 'fish',
                       'kept': kept
                       }
                odict.writerow(row)
    print('saving huber regreesion result to csv...Done')



folder_path = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=Jie%20Mei,volume=homes/rail/Middleton_2021'
video_names = sorted(os.listdir(folder_path))

for i, video_name in enumerate(tqdm(video_names)):
    if 'mp4' in video_name or 'PNG' in video_name:
        continue
    print('Huber Regression in video: ', video_name)

    ## huber regression
    file_path = '/home/jiemei/Documents/rail_detection/tracking_result_5_things_120_videos/Middleton_2021/'+video_name+'/tracking_result.csv'
    frame_path = os.path.join(folder_path, video_name)

    if i==0:
        det_ROI = np.load(file_path.replace(file_path.split('/')[-1], 'det_ROI.npy'))

    all_frame_names = np.load(file_path.replace('tracking_result.csv', 'detection_result_with_blank_frames_batch-all_frame_names.npy'),allow_pickle=True).tolist()

    data = load_mot_all_frames(file_path, all_frame_names)
    if len(data)==0:
        print('no detections in ', video_name)
        continue
    track_dict = read_track_dict(file_path, 'track id')

    direction = 'b2t'  # 'b2t',  represents the fish is pulled from bottom to top in the video. other choices are 'l2r', 'r2l', 't2b'
    missing_frame_num = 20
    data_huber = huber_regression(data, track_dict, direction, missing_frame_num)

    ## save as csv
    save_dir_csv = file_path.replace(file_path.split('/')[-1], 'tracking_result_with_huber.csv')
    save_data_huber_to_csv(save_dir_csv, data_huber, direction, det_ROI)

    ## save demo video
    save_dir = file_path.replace(file_path.split('/')[-1], 'demo')
    # save_as_one_video(frame_path, save_dir, data_huber, fps=20, box_thickness=3, txt_thickness=1, fontScale=0.5) # all frames/tracks will be saved in one video
    track_dict_huber = read_track_dict(save_dir_csv, 'id')
    
    if len(track_dict_huber)>10:
        print('save videos in ', video_name)
        save_as_videos(frame_path, save_dir, track_dict_huber, fps=20, box_thickness=3, txt_thickness=3,
                       fontScale=1)  # each track will be saves as one video
    else:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)










