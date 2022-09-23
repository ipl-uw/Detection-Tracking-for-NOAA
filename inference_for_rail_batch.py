from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from detectron2.engine import DefaultPredictor
from dataset_preprocess.train_test_split import rail_dataset_function
import random, cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from IPython import embed
import argparse
from tqdm import tqdm
import csv
import functools
from dataset_preprocess.xml_to_dict import category_ids
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import torch
thing_classes = list(category_ids.keys())


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def cmp_only_frame_id(s1,s2):

    # first 7 are id
    # print(s1)
    # print(s2)
    s1 = int(s1[:7])
    s2 = int(s2[:7])

    # last 5 are id
    # s1 = int(s1[-9:-4])
    # s2 = int(s2[-9:-4])

    if s1 < s2:
        return -1
    if s1 > s2:
        return 1
    return 0


def creat_csv(result_save_dir):
    f = open(result_save_dir+'/detection_result_with_blank_frames_batch.csv', 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame_id", "xmin", "ymin", "xmax", "ymax", "confidence","class"])

    return f, csv_writer

import detectron2.data.transforms as T
class Fish_Rail_Img(Dataset):
    """Only need tracking id, img for inference, but crop detected bbox"""

    def __init__(self, data_path,img_names,cfg=None):
        self.img_dir = data_path
        self.img_names = img_names
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )


    def __getitem__(self, index):

        img = cv2.imread(os.path.join(self.img_dir,self.img_names[index]))
        image = self.aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))


        img_name = self.img_names[index]


        return image, img_name, img.shape


    def __len__(self):
        return len(self.img_names)



### Loading Trained Faster RCNN Model###
print('Loading Faster RCNN Model...')
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("rail_train",)
# cfg.DATASETS.TEST = ()
cfg.OUTPUT_DIR = './output_'+str(len(thing_classes))+'_things_nonfish_sleeper_shark'
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.DEVICE = "cuda:1"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0484999.pth")  # path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0079999.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.DATASETS.TEST = ('rail_test',)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
# predictor = DefaultPredictor(cfg)


model = build_model(cfg) # returns a torch.nn.Module
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS) # must load weights this way, can't use cfg.MODEL.WEIGHTS = "..."
# model.train(False) # inference mode
model.eval()


data_path = './dataset_preprocess/rail_data/dataset_dicts.npz'
DatasetCatalog.register("rail_" + "test", lambda d="test":rail_dataset_function(data_path, mode=d))
MetadataCatalog.get("rail_" + "test").set(thing_classes=thing_classes)
rail_metadata = MetadataCatalog.get("rail_test")

print('Loading Faster RCNN Model... Done!')


### Run Model on Costum Dataset and Save CSV file ###
parser = argparse.ArgumentParser(description='Run Faster R-CNN Detector on Unlabeled Rail Data and Save Result as an CSV')
parser.add_argument('--result_save_dir', type=str,
                    default="./detection_result_"+str(len(thing_classes))+'_things',
                    help='It is the folder of detection result csv file')
parser.add_argument('--rail_data_path', type=str,
                    default="/run/user/1000/gvfs/smb-share:server=ipl-noaa.local,share=homes/rail/Predator_2018/20180824T162340-0800/GO-2400C-PGE+09-88-35",
                    help='It is the folder of rail data, the last folder should be the haul id, e.g. 20180602T112835-master')
args = parser.parse_args()


rail_data_path = args.rail_data_path
result_save_dir = os.path.join(args.result_save_dir, rail_data_path.split('/')[-3],rail_data_path.split('/')[-2])

visualization_dir = result_save_dir+'/visualization2_batch_test'
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)
    os.makedirs(visualization_dir)

f, csv_writer = creat_csv(result_save_dir)

# sort image names in order
BATCH_SIZE=20
image_names = sorted([f for f in os.listdir(rail_data_path) if f!="Thumbs.db"], key=functools.cmp_to_key(cmp_only_frame_id))
dataset = Fish_Rail_Img(data_path = rail_data_path,img_names=image_names, cfg=cfg)
data_loader = DataLoader(dataset=dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4,
                         pin_memory=True)
num=0
saved_num=0
all_rows = []
import time
with torch.no_grad():
    for (imgs, img_names, shapes) in tqdm(data_loader):

        inputs = []

        for ii, img in enumerate(imgs):
            inputs.append({'image':img,"height": shapes[0][ii], "width": shapes[1][ii]})

        # start=time.time()
        all_outputs = model(inputs)
        # end_time = time.time()
        # print(end_time-start)


        for j, outputs in enumerate(all_outputs):
            if len(outputs["instances"]) == 0:  ### no predicted objects ###
                # csv_writer.writerow([img_names[j], '', '', '', '', 0, ''])
                all_rows.append([img_names[j], '', '', '', '', 0, ''])
                continue
            # embed()

            max_score = 0
            for i in range(len(outputs["instances"])):

                xmin = outputs["instances"].pred_boxes.tensor[i][0]
                ymin = outputs["instances"].pred_boxes.tensor[i][1]
                xmax = outputs["instances"].pred_boxes.tensor[i][2]
                ymax = outputs["instances"].pred_boxes.tensor[i][3]
                score = outputs["instances"].scores[i]
                cls = outputs["instances"].pred_classes[i]

                # csv_writer.writerow([img_names[j], xmin, ymin, xmax, ymax, score, thing_classes[cls]])
                all_rows.append([img_names[j], xmin, ymin, xmax, ymax, score, thing_classes[cls]])

                if score > max_score:
                    max_score = score

            num+=1

# save time by Avoid unnecessary CPU-GPU synchronization
for row in all_rows:
    if '' not in row: # need convert from tensor to number
        row=[row[0], row[1].item(), row[2].item(0), row[3].item(), row[4].item(), row[5].item(), row[6].item()]
    csv_writer.writerow(row)

f.close()
print('Detect %d frames with objects in haul %s'%(num, rail_data_path[rail_data_path.rfind('/')+1:]))