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

thing_classes = list(category_ids.keys())

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
    f = open(result_save_dir+'/detection_result_with_blank_frames.csv', 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame_id", "xmin", "ymin", "xmax", "ymax", "confidence","class"])

    return f, csv_writer


### Loading Trained Faster RCNN Model###
print('Loading Faster RCNN Model...')
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("rail_train",)
# cfg.DATASETS.TEST = ()
cfg.OUTPUT_DIR = './output_'+str(len(thing_classes))+'_things_sleeper_nonfish'
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0484999.pth")  # path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0079999.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.DATASETS.TEST = ('rail_test',)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
predictor = DefaultPredictor(cfg)

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
                    default="/run/user/1000/gvfs/smb-share:server=ipl-noaa.local,share=homes/rail/Predator_2018/20180824T194439-0800/GO-2400C-PGE+09-88-35",
                    help='It is the folder of rail data, the last folder should be the haul id, e.g. 20180602T112835-master')
args = parser.parse_args()


rail_data_path = args.rail_data_path
result_save_dir = os.path.join(args.result_save_dir, rail_data_path.split('/')[-3],rail_data_path.split('/')[-2])

visualization_dir = result_save_dir+'/visualization2'
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)
    os.makedirs(visualization_dir)

f, csv_writer = creat_csv(result_save_dir)

# sort image names in order
image_names = sorted([f for f in os.listdir(rail_data_path) if f!="Thumbs.db"], key=functools.cmp_to_key(cmp_only_frame_id))



num = 0
saved_num=0
for img_name in tqdm(image_names):
    if 'jpg' not in img_name and 'png' not in img_name:  # skip non image
        continue

    im = cv2.imread(os.path.join(rail_data_path, img_name))

    if im is None:  # skip bad frame
        print('bad frame: %s' %os.path.join(rail_data_path, img_name))
        continue

    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    if len(outputs["instances"])==0:  ### no predicted objects ###
        csv_writer.writerow([img_name, '', '', '', '', 0, ''])
        continue
    # embed()


    max_score = 0
    for i in range(len(outputs["instances"])):
        xmin = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()[i][0]
        ymin = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()[i][1]
        xmax = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()[i][2]
        ymax = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()[i][3]
        score = outputs["instances"].scores[i].item()
        cls = outputs["instances"].pred_classes[i].item()

        csv_writer.writerow([img_name,xmin,ymin, xmax, ymax,score, thing_classes[cls]])

        if score > max_score:
            max_score=score

    if max_score >=0.95 and saved_num<=100: # only save 100 HIGH conf samples
        v = Visualizer(im[:, :, ::-1],
                       metadata=rail_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAdataset_preprocessGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        # print(outputs['instances'].pred_classes)
        # print(outputs["instances"].pred_boxes)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(visualization_dir, img_name), out.get_image()[:, :, ::-1])
        saved_num+=1


    num+=1
    # cv2.imshow('inference sample',out.get_image()[:, :, ::-1])
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()

f.close()
print('Detect %d frames with objects in haul %s'%(num, rail_data_path[rail_data_path.rfind('/')+1:]))