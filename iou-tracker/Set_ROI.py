import cv2, imutils
import os
from IPython import embed


def Set_ROI(frame_path):
    '''
    
    Args:
        frame_path: 

    Returns:x, y, w, h, xy are top left corner.

    '''

    ### Step-1: read detection ROI given by user ###
    img = cv2.imread(os.path.join(frame_path,os.listdir(frame_path)[0]))
    (h, w) = img.shape[:2]

    width = 1000
    r = width / float(w)
    re_img = imutils.resize(img, width=1000)
    re_roi = cv2.selectROI(windowName="roi", img=re_img, showCrosshair=True, fromCenter=False)


    detection_roi = (re_roi[0]/r, re_roi[1]/r, re_roi[2]/r, re_roi[3]/r)
    x, y, w, h = detection_roi

    # draw = cv2.rectangle(img=img, pt1=(int(x), int(y)), pt2=(int(x + w), int(y + h)), color=(0, 0, 255), thickness=2)
    # cv2.imshow("roi", draw)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()

    return x, y, w, h


