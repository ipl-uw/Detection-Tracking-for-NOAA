#!/usr/bin/env python

# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------

import argparse

from iou_tracker import track_iou
from util import load_mot, save_to_csv
from viou_tracker import track_viou
from IPython import embed
import numpy as np
from Set_ROI import Set_ROI
import gc, os


def main(args, ROI):
    formats = ['motchallenge', 'visdrone', 'fish']
    assert args.format in formats, "format '{}' unknown supported formats are: {}".format(args.format, formats)

    with_classes = False
    if args.format == 'visdrone':
        with_classes = True

    if args.read_npy:
        detections = np.load(os.path.join(args.output_path, args.detection_path.split('/')[-1].replace('csv', 'npy')),allow_pickle=True).tolist()
        all_frame_name = np.load(os.path.join(args.output_path, args.detection_path.split('/')[-1].replace('.csv', '-all_frame_names.npy')),allow_pickle=True).tolist()
        # detections = np.load(args.output_path.replace(args.output_path.split('/')[-1], args.detection_path.split('/')[-1].replace('csv', 'npy')), allow_pickle=True).tolist()
        # all_frame_name = np.load(args.output_path.replace(args.output_path.split('/')[-1], args.detection_path.split('/')[-1].replace('.csv', '-all_frame_names.npy')), allow_pickle=True).tolist()
        print('read saved npy done!')
    else:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        detections, all_frame_name = load_mot(args.detection_path, nms_overlap_thresh=args.nms, with_classes=True)
        np.save(args.output_path+'/'+ args.detection_path.split('/')[-1].replace('csv', 'npy'), detections)
        np.save(args.output_path+ '/'+args.detection_path.split('/')[-1].replace('.csv', '-all_frame_names.npy'), all_frame_name)




    if args.visual:
        tracks = track_viou(args.frames_path, detections, args.sigma_l, args.sigma_h, args.sigma_iou, args.t_min,
                            args.ttl, args.visual, args.keep_upper_height_ratio, ROI, args.track_cls)
    else:
        if with_classes:
            # track_viou can also be used without visual tracking, but note that the speed will be much slower compared
            # to track_iou. However, this way supports the optimal LAP solving and the handling of multiple object classes:
            tracks = track_viou(args.frames_path, detections, args.sigma_l, args.sigma_h, args.sigma_iou, args.t_min,
                                args.ttl, 'NONE', args.keep_upper_height_ratio)
        else:
            tracks = track_iou(detections, args.sigma_l, args.sigma_h, args.sigma_iou, args.t_min, ROI, args.track_cls)
    print('Total %d tracks.' %len(tracks))
    save_to_csv(args.output_path, tracks, all_frame_name, fmt=args.format)

from tqdm import tqdm
if __name__ == '__main__':

    folder_path = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=Jie%20Mei,volume=homes/rail/Middleton_2021'
    video_names = sorted(os.listdir(folder_path))

    for i, video_name in enumerate(tqdm(video_names)):
        if 'mp4' in video_name or 'PNG' in video_name:
            continue
        print('Tracking in video: ', video_name)

        parser = argparse.ArgumentParser(description="IOU/V-IOU Tracker demo script")




        parser.add_argument('-v', '--visual', type=str, default='', help="visual tracker for V-IOU. Currently supported are "
                                                             "[BOOSTING, MIL, KCF, KCF2, TLD, MEDIANFLOW, GOTURN, NONE] "
                                                             "see README.md for furthert details")
        parser.add_argument('-hr', '--keep_upper_height_ratio', type=float, default=1.0,
                            help="Ratio of height of the object to track to the total height of the object "
                                 "for visual tracking. e.g. upper 30%%")
        parser.add_argument('-sl', '--sigma_l', type=float, default=0.4,
                            help="low detection threshold")
        parser.add_argument('-sh', '--sigma_h', type=float, default=0.8,
                            help="high detection threshold")
        parser.add_argument('-si', '--sigma_iou', type=float, default=0.1,
                            help="intersection-over-union threshold")
        parser.add_argument('-tm', '--t_min', type=float, default=15,
                            help="minimum track length")
        parser.add_argument('-ttl', '--ttl', type=int, default=10,
                            help="time to live parameter for v-iou")
        parser.add_argument('-nms', '--nms', type=float, default=0.2,
                            help="nms for loading multi-class detections")
        parser.add_argument('-fmt', '--format', type=str, default='fish',
                            help='format of the detections [motchallenge, visdrone, fish]')
        parser.add_argument('-read_npy', '--read_npy', type=str, default=False,
                            help='read saved npy file, or read from csv file')
        parser.add_argument('-track_cls', '--track_cls', type=list, default=['Fish'],
                            help='target objects, e.g. [\'Fish\', \'Bait\',\'Live Birds\',\'Fishing Gear\',\'Dead Birds\']')


        args1 = parser.parse_args()

        args1.frames_path = os.path.join(folder_path, video_name)
        args1.detection_path = os.path.join('/home/jiemei/Documents/rail_detection/detection_result_5_things_120_videos', video_name,'detection_result_with_blank_frames_batch.csv')
        args1.output_path = '/home/jiemei/Documents/rail_detection/tracking_result_5_things_120_videos/Middleton_2021/' + video_name


        assert not args1.visual or args1.visual and args1.frames_path, "visual tracking requires video frames, " \
                                                                    "please specify via --frames_path"

        assert 0.0 < args1.keep_upper_height_ratio <= 1.0, "only values between 0 and 1 are allowed"
        assert args1.nms is None or 0.0 <= args1.nms <= 1.0, "only values between 0 and 1 are allowed"

        if not os.path.exists(args1.output_path):
            os.makedirs(args1.output_path)

        if i==0:
            (x, y, w, h) = Set_ROI(args1.frames_path)
            np.save(args1.output_path + '/det_ROI.npy', (x, y, w, h))


        main(args1, (x,y,w,h))





