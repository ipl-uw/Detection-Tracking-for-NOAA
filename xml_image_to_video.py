import os
from IPython import embed
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm

def findIdTmp(name):
    i = len(name)
    while name[i - 1] in '0123456789':#一直把i 往前挪动，指向非数字的最右一个位置
        i -= 1

    #如果名字里没有‘0123456789’，则返回 tmpid -1
    if i == len(name):
        return (name, -1) #default id
    else: #如果名字里有数字，现在i指向最后一个非数字的位置
        j = i
        while name[j - 1] in '_':  #让j指向非 ‘_’的最后一个位置
            j -= 1
        return (name[:j], int(name[i:]))  #返回的名字里没有下划线，同时返回tempid


xml_path ='/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=Jie%20Mei,volume=homes/rail/Predator_2018/20180824T162340-0800/labels_Pacific Sleeper Sharks'
images_path = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=Jie%20Mei,volume=homes/rail/Predator_2018/20180824T162340-0800/GO-2400C-PGE+09-88-35'
save_dir = xml_path.replace(xml_path.split('/')[-1], 'shark.avi')

files = os.listdir(xml_path)
files.sort()

color = (0, 255, 0)
fontScale = 1
txt_thickness = 1
fps = 30
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
box_thickness=3
begining_frame = 14500
num_frames = 500
for i, xml in tqdm(enumerate(files)):
    if i<begining_frame:
        continue
    tree = ET.parse(os.path.join(xml_path, xml))
    root = tree.getroot()
    if root.findall('object') == None:
        print('None object in xml' + os.path.join(xml_path, xml))
        continue


    filename = root.find('filename').text
    im = cv2.imread(os.path.join(images_path,filename))

    size =root.findall('size')[0]

    if i==begining_frame:
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        videoWriter = cv2.VideoWriter(save_dir, fourcc, fps, (width,height))  # (1360,480)为视频大小



    # 读取当前xml里 所有的object
    objs = []
    for obj in root.findall('object'):
        name = obj.find('name').text

        species, id_tmp = findIdTmp(name)  # 去掉下划线 之后的所有字符

        # 找到box的数据
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        # print(name+': {0}, {1}, {2}, {3}'.format(xmin, ymin, xmax, ymax))

        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, box_thickness)
        cv2.putText(im, species, (xmin, ymax+25), cv2.FONT_HERSHEY_COMPLEX,fontScale, color, txt_thickness)
        videoWriter.write(im)
    if i > begining_frame + num_frames:
         break

videoWriter.release()

