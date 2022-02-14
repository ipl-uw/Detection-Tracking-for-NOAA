import os
from dataset_preprocess.xml_to_dict import xml_to_dict
import xml.etree.ElementTree as ET
from IPython import embed
from tqdm import tqdm
import cv2
import time
import numpy as np
import random


def generate_rail_dataset(label_dir, sub_folder):
    print('generating dataset for detection...')
    img_dir = label_dir.replace('labels', 'images')

    dataset_dicts = []
    non_images_haul = ['20181102T135400-0800']  # yes label, no frames

    ships = os.listdir(label_dir)
    idx = 1

    # data distribution histogram #
    original_sp_count = {}
    Alias_sp_count = {}

    Not_found_alias={}
    for ship in tqdm(ships):
        hauls = os.listdir(os.path.join(label_dir,ship))
        for haul in hauls:
            if haul in non_images_haul:  # 跳过没有images的haul
                print('skip ' + haul)
                continue
            # if haul != '20180618T093255-0800':
            #     continue


            # 读取所有xml文件
            # time_start = time.time()
            xml_path = os.path.join(label_dir, ship, haul,'labels')
            xmls = [f for f in os.listdir(xml_path) if f[-4:] == '.xml']
            xmls.sort()


            img_path = os.path.join(img_dir,ship,haul)
            # time_end = time.time()
            # print('time cost', time_end - time_start, 's')
            # embed()


            for xml in xmls:
                #for each label use a dic to restore the annotations for each image

                tree = ET.parse(os.path.join(xml_path, xml))
                root = tree.getroot()
                if root.findall('object') == None:
                    print('None object in xml' + os.path.join(xml_path, xml))
                    continue

                record, original_sp_count, Alias_sp_count, idx, Not_found_alias= xml_to_dict(img_path, root, original_sp_count, Alias_sp_count,idx, Not_found_alias)

                if record!={}: # somtimes, no fish objects in xml file, then got a blank record
                    dataset_dicts.append(record)


    print('Not found alias: ', Not_found_alias)
    ### top 90% for train, last 10% for test ###
    random.shuffle(dataset_dicts)
    np.savez('./dataset_preprocess/'+sub_folder+'/original_sp_count.npz', original_sp_count)
    np.savez('./dataset_preprocess/'+sub_folder+'/Alias_sp_count.npz', Alias_sp_count)
    np.savez('./dataset_preprocess/'+sub_folder+'/dataset_dicts.npz', dataset_dicts)

    print('generating dataset for detection...Done!')
    return dataset_dicts #len(dataset_dicts) is num of labeled frames, may be less than Alias_sp_count because multiple fish in one frame

if __name__ == '__main__':
    # label_path = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=Jie%20Mei,volume=home/rail data/labels'

    # save folder for labels
    label_path = '/run/user/1000/gvfs/smb-share:server=ipl-noaa.local,share=homes/Jie Mei/sleeper shark data chain/labels'
    # represents the feature of this new data
    sub_folder = 'only_sleeper_shark_plus_chain'
    dataset_dicts = generate_rail_dataset(label_path, sub_folder)