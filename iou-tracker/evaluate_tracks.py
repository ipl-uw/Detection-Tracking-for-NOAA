# Evaluate tracking performance
# True positive (TP) counts
# False negative (FN) counts
# False positive (FP) counts

import sys
import csv
import numpy as np


# def main(argv):
#     if len(argv) < 4:
#         print('[ERROR] Not enough arguments')
#         print('Examples:')
#         print('filename_gt filename_pred filename_out')
#         print('or')
#         print('filename_gt filename_pred filename_out method')
#         print('method can be simple, greedy, or mbp (default: simple)')
#         return -1
#     if len(argv) == 4:
#         return evaluate(argv[1], argv[2], argv[3])
#     else:
#         return evaluate(argv[1], argv[2], argv[3], argv[4])



class Track:
    def __init__(self, id, t, box):
        self.id = id
        self.t_first = t
        self.t_last = t
        self.boxes = [box] # list of (t, xmin, ymin, xmax, ymax)
        self.matched = False # whether matched with gt/pred

    def __str__(self):
        return '%d: (%d, %d), %d, %s' % (self.id, self.t_first, self.t_last, len(self.boxes), self.matched)


def iou(box1, box2):
    '''
    Calculate intersection-over-union (IOU) between box1 and box2
    Inputs:
        box1 and box2: (xmin, ymin, xmax, ymax)
    Returns:
        IOU
    '''
    xmin1 = box1[0]
    ymin1 = box1[1]
    xmax1 = box1[2]
    ymax1 = box1[3]
    xmin2 = box2[0]
    ymin2 = box2[1]
    xmax2 = box2[2]
    ymax2 = box2[3]
    # box1 and box2 must satisfy xmin <= xmax and ymin <= ymax !

    dx = min(xmax1, xmax2) - max(xmin1, xmin2)
    if dx <= 0:
        return 0.0
    dy = min(ymax1, ymax2) - max(ymin1, ymin2)
    if dy <= 0:
        return 0.0
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    return dx*dy / float(area1 + area2 - dx*dy)


def overlap_counts(gt, pred, iou_thresh):
    overlap = 0
    idx_gt = 0
    idx_pred = 0
    while idx_gt < len(gt.boxes) and idx_pred < len(pred.boxes):
        if gt.boxes[idx_gt][0] < pred.boxes[idx_pred][0]:
            idx_gt += 1
        elif pred.boxes[idx_pred][0] < gt.boxes[idx_gt][0]:
            idx_pred += 1
        else:
            if iou(pred.boxes[idx_pred][1:], gt.boxes[idx_gt][1:]) > iou_thresh:
                overlap += 1
            idx_gt += 1
            idx_pred += 1
    return overlap


# Python program to find
# maximal Bipartite matching
# [Ref] https://www.geeksforgeeks.org/maximum-bipartite-matching/
class GFG: 
    def __init__(self, graph):
        # residual graph
        self.graph = graph
        self.num_gt = graph.shape[0]
        self.num_pred = graph.shape[1]
        self.match= None

    # A DFS based recursive function
    # that returns true if train matching
    # for vertex u is possible
    def bpm(self, u, matchR, seen):
        for v in range(self.num_pred):
            if self.graph[u][v] and seen[v] == False:
                # Mark v as visited
                seen[v] = True
                if matchR[v] == -1 or self.bpm(matchR[v], matchR, seen):
                    matchR[v] = u
                    return True
        return False

    # Returns maximum number of matching
    def maxBPM(self):
        matchR = [-1] * self.num_pred
        result = 0 
        for i in range(self.num_gt):
            seen = [False] * self.num_pred
            if self.bpm(i, matchR, seen):
                result += 1
        self.match = matchR
        return result


def evaluate(filename_gt, filename_pred, filename_out=None, method='simple', iou_thresh=0.5, overlap_thresh=0.1, gt_thresh=20, pred_thresh=10):
    '''
    Evaluate tracking performance
    Inputs:
        filename_gt: ground truth csv filename
        filename_pred: predicted csv filename
        filename_out: output filename
        iou_thresh: threshold of intersection-over-union (IOU)
        overlap_thresh: threshold of overlapping in time
    Returns:
        TP: true positive counts
        FN: false negative counts
        FP: false positive counts
    '''
    # Read ground truth and prediction
    print('Loading gt file: %s'%filename_gt)
    tracks_gt = read_tracks(filename_gt)
    print('Loading pred file: %s'%filename_pred)
    tracks_pred = read_tracks(filename_pred)
    print('Number of gt: %d'%len(tracks_gt))
    print('Number of pred: %d'%len(tracks_pred))

    # Remove tracks which are shorter than threshold
    print('gt_thresh = %d' % gt_thresh)
    print('pred_thresh = %d' % pred_thresh)
    tracks_gt = [t for t in tracks_gt if len(t.boxes) >= gt_thresh]
    tracks_pred = [t for t in tracks_pred if len(t.boxes) >= pred_thresh]
    print('Number of gt after thresholding: %d'%len(tracks_gt))
    print('Number of pred after thresholding: %d'%len(tracks_pred))

    # Match ground truth to prediction
    print('iou_thresh: %f'%iou_thresh)
    print('overlap_thresh: %f'%overlap_thresh)
    map_gt = dict()
    map_pred = dict()
    if method == 'simple':
        # Simple match
        print('Using method simple')
        for gt in tracks_gt:
            for pred in tracks_pred:
                # print('%d %d' % (gt.id, pred.id))
                if pred.matched or gt.matched:
                    continue
                if pred.t_last < gt.t_first or pred.t_first > gt.t_last:
                    continue
                overlap = overlap_counts(gt, pred, iou_thresh)


                # if gt.id ==90037:
                #     from IPython import embed
                #     embed()

                    # print(pred.id)
                    # print(overlap)

                # from IPython import embed
                # embed()

                if overlap >= overlap_thresh*len(gt.boxes):
                    map_gt[gt.id] = pred.id
                    map_pred[pred.id] = gt.id
                    gt.matched = True
                    pred.matched = True
    elif method == 'greedy':
        print('Using method greedy')
        # Greedy match
        num_gt = len(tracks_gt)
        num_pred = len(tracks_pred)
        match_mat = np.zeros((num_gt, num_pred))
        for i_gt in range(num_gt):
            for i_pred in range(num_pred):
                gt = tracks_gt[i_gt]
                pred = tracks_pred[i_pred]
                if pred.t_last < gt.t_first or pred.t_first > gt.t_last:
                    continue
                overlap = overlap_counts(gt, pred, iou_thresh)
                match_mat[i_gt, i_pred] = overlap/float(len(gt.boxes))
        ind = np.unravel_index(np.argsort(match_mat, axis=None)[::-1], match_mat.shape)
        for i in range(num_gt*num_pred):
            i_gt = ind[0][i]
            i_pred = ind[1][i]
            if match_mat[i_gt, i_pred] < overlap_thresh:
                break
            gt = tracks_gt[i_gt]
            pred = tracks_pred[i_pred]
            if gt.matched or pred.matched:
                continue
            map_gt[gt.id] = pred.id
            map_pred[pred.id] = gt.id
            gt.matched = True
            pred.matched = True
    elif method == 'mbp':
        # Dynamic programming
        print('Using method mbp')
        num_gt = len(tracks_gt)
        num_pred = len(tracks_pred)
        match_mat = np.zeros((num_gt, num_pred))
        for i_gt in range(num_gt):
            for i_pred in range(num_pred):
                gt = tracks_gt[i_gt]
                pred = tracks_pred[i_pred]
                if pred.t_last < gt.t_first or pred.t_first > gt.t_last:
                    continue
                overlap = overlap_counts(gt, pred, iou_thresh)
                match_mat[i_gt, i_pred] = overlap/float(len(gt.boxes))
        bpGraph = match_mat >= overlap_thresh
        g = GFG(bpGraph)
        g.maxBPM()
        for i_pred in range(num_pred):
            i_gt = g.match[i_pred]
            if i_gt >= 0:
                gt = tracks_gt[i_gt]
                pred = tracks_pred[i_pred]
                map_gt[gt.id] = pred.id
                map_pred[pred.id] = gt.id
                gt.matched = True
                pred.matched = True
    else:
        raise ValueError('Name of method unknown %s' % method)

    # Calculate TP, FN, FP
    TP = len(map_gt)
    FP = len(tracks_pred) - TP
    FN = len(tracks_gt) - TP

    # Write result to file
    if filename_out:
        with open(filename_out, 'w') as f:
            f.write('#TP: %d\n'%TP)
            f.write('#FN: %d\n'%FN)
            f.write('#FP: %d\n'%FP)
            # TP
            for gt in tracks_gt:
                if gt.id in map_gt:
                    f.write('%d,%d\n'%(gt.id, map_gt[gt.id]))
            # FN
            for gt in tracks_gt:
                if gt.id not in map_gt:
                    f.write('%d,%d\n'%(gt.id, -1))
            # FP
            for pred in tracks_pred:

                # from IPython import embed
                # embed()

                if pred.id not in map_pred:
                    f.write('%d,%d\n'%(-1, pred.id))

    print('TP: %d'%TP)
    print('FN: %d'%FN)
    print('FP: %d'%FP)

    return TP, FN, FP


def read_tracks(filename, col_id=0, col_frame=1, col_xmin=2, col_ymin=3, col_xmax=4, col_ymax=5, frame_digit=7, skip_header=True):
    '''
    Read tracks from csv file and sort
    Returns:
        A sorted list of tracks
    '''
    # Read csv into dictionary
    dict_tracks = dict()
    is_header = skip_header
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if is_header:
                is_header = False
                continue
            # print(row[col_id])
            id = int(row[col_id])



            # print(row[col_frame][:frame_digit])
            # t = int(row[col_frame][:frame_digit])
            t = int(row[col_frame][-9:-4])


            #one fish is always floating in water, so remove it, in haul 43
            if t>4280:
                continue





            xmin = float(row[col_xmin])
            ymin = float(row[col_ymin])
            xmax = float(row[col_xmax])
            ymax = float(row[col_ymax])
            if id not in dict_tracks:
                dict_tracks[id] = Track(id, t, (t, xmin, ymin, xmax, ymax))
            else:
                dict_tracks[id].boxes.append((t, xmin, ymin, xmax, ymax))
                dict_tracks[id].t_last = t
    # Make sorted list of track by time
    tracks = list(dict_tracks.values())
    for track in tracks:
        track.boxes.sort(key=lambda x: x[0])
        track.t_first = track.boxes[0][0]
        track.t_last = track.boxes[-1][0]
    tracks.sort(key=lambda x: x.t_first)

    # Print for debug
    #for track in tracks:
    #    print(track)

    return tracks


if __name__ == '__main__':

    gt_path = '/home/jiemei/Documents/rail_detection/data/Middleton_IP_2020/1_2020-07-16_21-43-03 (2-5-2021 10-03-51 AM)/hierarchy_labels_2nd_level.csv'
    predict_path ='/home/jiemei/Documents/rail_detection/tracking_result_5_things/Middleton_IP_2020/1_2020-07-16_21-43-03 (2-5-2021 10-03-51 AM)/tracking_result_with_huber.csv'
    evaluate(gt_path, predict_path)