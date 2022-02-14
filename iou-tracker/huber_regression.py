
import numpy as np
from viou_tracker import in_ROI
from sklearn.linear_model import HuberRegressor
from IPython import embed
from tqdm import tqdm


def direction_judge(track, previous_track, direction):

    first_frame_info = track[0]
    last_frame_info = previous_track[-1]

    first_frame_bbox_xc = (float(first_frame_info['xmin']) + float(first_frame_info['xmax']))/2
    first_frame_bbox_yc = (float(first_frame_info['ymin']) + float(first_frame_info['ymax'])) / 2
    last_frame_bbox_xc = (float(last_frame_info['xmin']) + float(last_frame_info['xmax'])) / 2
    last_frame_bbox_yc = (float(last_frame_info['ymin']) + float(last_frame_info['ymax'])) / 2

    second_frame_info = track[4]
    last_2nd_frame_info = previous_track[-5]

    second_frame_bbox_xc = (float(second_frame_info['xmin']) + float(second_frame_info['xmax'])) / 2
    second_frame_bbox_yc = (float(second_frame_info['ymin']) + float(second_frame_info['ymax'])) / 2
    last_2nd_frame_bbox_xc = (float(last_2nd_frame_info['xmin']) + float(last_2nd_frame_info['xmax'])) / 2
    last_2nd_frame_bbox_yc = (float(last_2nd_frame_info['ymin']) + float(last_2nd_frame_info['ymax'])) / 2



    if direction =='l2r':
        if first_frame_bbox_xc > last_frame_bbox_xc:
            flag = True
        else:
            flag  =False

        direction_track = second_frame_bbox_xc- first_frame_bbox_xc
        direction_previous_track = last_frame_bbox_xc - last_2nd_frame_bbox_xc

        flag = (direction_track*direction_previous_track>0) and flag  # same direction and flag


    elif direction =='r2l':
        if first_frame_bbox_xc < last_frame_bbox_xc:
            flag = True
        else:
            flag  =False

        direction_track = second_frame_bbox_xc - first_frame_bbox_xc
        direction_previous_track = last_frame_bbox_xc - last_2nd_frame_bbox_xc

        flag = (direction_track * direction_previous_track > 0) and flag

    elif direction =='b2t':


        if first_frame_bbox_yc < last_frame_bbox_yc:
            flag = True
        else:
            flag  =False

        direction_track = second_frame_bbox_yc - first_frame_bbox_yc
        direction_previous_track = last_frame_bbox_yc - last_2nd_frame_bbox_yc

        flag = (direction_track * direction_previous_track > 0) and flag

    elif direction =='t2b':
        if first_frame_bbox_yc > last_frame_bbox_yc:
            flag = True
        else:
            flag  =False

        direction_track = second_frame_bbox_yc - first_frame_bbox_yc
        direction_previous_track = last_frame_bbox_yc - last_2nd_frame_bbox_yc

        flag = (direction_track * direction_previous_track > 0) and flag

    else:
        print('NA direction: %s, please use one of [l2r, r2l, b2t, t2b]' %direction)



    return flag





def huber_regression(data, track_dict, direction, missing_frame_num):
    '''

    Args:
        data: [[{},{},..], [], []...] one list represents one frame, one dict represents one detections in this frame, data has all frames inside, if there is no
    detection in one frame, then [{'id': '', 'bbox': ('', '','','',), 'score': '', 'class': '','frame_name': frame_name}]
        track_dict: { 'track_id': [{},{},{}], :[],...}, according to track id, put all corresponding frames info into a list
        direction: one of ['l2r', 'r2l', 'b2t', 't2b']

    Returns: data_huber, use huber regression to interpolate missing detections between two broken tracks. and merge these 2 tracks into one track

    '''
    data_huber = data.copy()

    print('Doing huber regression...')

    for i, track_id in tqdm(enumerate(track_dict)):


        track = track_dict[track_id]


        if i==0:
            previous_tracks = [track]
            continue

        first_frame_info = track[0]
        first_frame_id = int(first_frame_info['frame'].split('_')[1].split('.')[0])         # standard frame id are first several digits and followed by '_'
        # first_frame_id = int(first_frame_info['frame'].split(' ')[-1][:5])  # middleton 2020 frame id format is different!! 58-04
        # first_frame_id = int(first_frame_info['frame'].split(' ')[-1][:4])  # middleton 2020 frame id format is different!! 43-03



        for j, previous_track in enumerate(previous_tracks):
            last_frame_info = previous_track[-1]
            last_frame_id = int(last_frame_info['frame'].split('_')[1].split('.')[0]) # standard frame id are first several digits and followed by '_'
            # last_frame_id = int(last_frame_info['frame'].split(' ')[-1][:5])
            # last_frame_id = int(last_frame_info['frame'].split(' ')[-1][:4])


            direction_satisfy = direction_judge(track, previous_track, direction)



            if missing_frame_num>=first_frame_id - last_frame_id>0 and direction_satisfy: # connect consecutive tracks!
            # if missing_frame_num >= first_frame_id - last_frame_id > 0:  # connect!

                xmin = []
                ymin = []
                xmax = []
                ymax = []
                for frame_info in previous_track:
                    xmin.append(float(frame_info['xmin']))
                    ymin.append(float(frame_info['ymin']))
                    xmax.append(float(frame_info['xmax']))
                    ymax.append(float(frame_info['ymax']))

                #1. modify track id in data_huber,  make later tracks' track id# the same as previous track
                for frame in data_huber:
                    for det in frame:
                        if det['id'] == int(track_id):
                            det['id'] = int(previous_track[0]['track id'])
                            xmin.append(det['bbox'][0])
                            ymin.append(det['bbox'][1])
                            xmax.append(det['bbox'][2])
                            ymax.append(det['bbox'][3])


                # 1.1 modify track id in current track, so that previous_track = track, can get new id, for next loop  (if merge again r.g. 8-9-10 all change to 8), but next track merge only use this track for huber regression instead of merged track
                for frame in track:
                    frame['track id']= int(previous_track[0]['track id'])


                #2. huber regression missing detections, add bbox into data_huber
                if first_frame_id - last_frame_id== 1:  # no missing detection
                    previous_tracks.append(track)
                    if len(previous_tracks) == 1:
                        previous_tracks.pop(0)
                    continue
            
                t = np.arange((int(track[-1]['frame'].split('_')[1].split('.')[0])-int(previous_track[0]['frame'].split('_')[1].split('.')[0])+1)).reshape(-1,1)   # standard frame id are first several digits and followed by '_'
                # t = np.arange((int(track[-1]['frame'].split(' ')[-1][:5]) - int(previous_track[0]['frame'].split(' ')[-1][:5]) + 1)).reshape(-1, 1)
                # t = np.arange((int(track[-1]['frame'].split(' ')[-1][:4]) - int(previous_track[0]['frame'].split(' ')[-1][:4]) + 1)).reshape(-1,1)




                train_t = np.concatenate((t[:len(previous_track)] , t[-len(track):]))
                test_t = t[len(previous_track): len(previous_track)+ len(t)-len(train_t)]


                huber_xmin = HuberRegressor(alpha=0.0001).fit(train_t, np.array(xmin))
                huber_ymin = HuberRegressor().fit(train_t, np.array(ymin))
                huber_xmax = HuberRegressor().fit(train_t, np.array(xmax))
                huber_ymax = HuberRegressor().fit(train_t, np.array(ymax))

                regress_xmin = huber_xmin.predict(test_t)
                regress_ymin = huber_ymin.predict(test_t)
                regress_xmax = huber_xmax.predict(test_t)
                regress_ymax = huber_ymax.predict(test_t)

                count=0
                for frame in data_huber:
                    if last_frame_id<int(frame[0]['frame_name'].split('_')[1].split('.')[0]) < first_frame_id:   # standard frame id are first several digits and followed by '_'
                    # if last_frame_id < int(frame[0]['frame_name'].split(' ')[-1][:5]) < first_frame_id:
                    # if last_frame_id < int(frame[0]['frame_name'].split(' ')[-1][:4]) < first_frame_id:
                        if frame[0]['id']=='':
                            frame[0]['id'] = int(previous_track[0]['track id'])
                            frame[0]['bbox'] = (regress_xmin[count], regress_ymin[count], regress_xmax[count], regress_ymax[count])
                            frame[0]['score'] = -2
                            frame[0]['class'] = 'fish'

                        else:
                            frame.append({'id': int(previous_track[0]['track id']), 'bbox':(regress_xmin[count], regress_ymin[count], regress_xmax[count], regress_ymax[count]), 'score':-2, 'class':'fish', 'frame_name': frame[0]['frame_name']})
                        count+=1

                    if count== len(regress_xmax): # quick break because finish all missing detections
                        break
                previous_tracks.pop(j)
                break


        previous_tracks.append(track)
        if len(previous_tracks)==3:
            previous_tracks.pop(0)






    print('Doing huber regression...Done')


    return data_huber




