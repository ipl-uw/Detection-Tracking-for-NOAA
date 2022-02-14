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
from dataset_preprocess.xml_to_dict import category_ids

thing_classes = list(category_ids.keys())

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("rail_train",)
# cfg.DATASETS.TEST = ()
# cfg.OUTPUT_DIR = './output_'+str(len(thing_classes))+'_things'
# cfg.OUTPUT_DIR = './output_'+str(len(thing_classes))+'_things'+'fintune_on_sleeper_shark_continue2'
cfg.OUTPUT_DIR = './output_'+str(len(thing_classes))+'_things'+'fintune_on_sleeper_shark_plus_chain_continue'


cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)   # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0474999.pth")  # path to the model we just trained
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0044999.pth")  # path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0104999.pth")  # path to the model we just trained

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.DATASETS.TEST = ('rail_test',)

predictor = DefaultPredictor(cfg)

# data_path = './dataset_preprocess/dataset_dicts.npz'
# data_path = './dataset_preprocess/only_sleeper_shark/dataset_dicts.npz'
data_path = './dataset_preprocess/only_sleeper_shark_plus_chain/dataset_dicts.npz'

### randomly select several samples to visualize the prediction results. ###
for d in ["train", "test"]:
    DatasetCatalog.register("rail_" + d, lambda d=d:rail_dataset_function(data_path, mode=d))
    MetadataCatalog.get("rail_" + d).set(thing_classes=thing_classes)
rail_metadata = MetadataCatalog.get("rail_test")


dataset_dicts = rail_dataset_function(data_path, 'test')
# sample_save_dir = "./output_eva_"+str(len(thing_classes))+'_things'
# sample_save_dir = "./output_eva_"+str(len(thing_classes))+'_things_sleeper_shark'
sample_save_dir = "./output_eva_"+str(len(thing_classes))+'_things_sleeper_shark_plus_chain_continue'

if not os.path.exists(sample_save_dir):
    os.makedirs(sample_save_dir)

for d in random.sample(dataset_dicts, 50):
# for d in dataset_dicts:
#     if '20180618T093255-0800' not in d["file_name"]:
#         continue

# aukbay2018_path = '/run/user/1000/gvfs/smb-share:server=ipl-noaa.local,share=home/some example for test faster rcnn/20180602T112835-master'
# for img_name in os.listdir(aukbay2018_path):

    im = cv2.imread(d["file_name"])
    # im = cv2.imread(os.path.join(aukbay2018_path,img_name ))
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=rail_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    print(outputs['instances'].pred_classes)
    print(outputs["instances"].pred_boxes)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # embed()
    cv2.imwrite(sample_save_dir+d["file_name"][d["file_name"].rfind('/'):],out.get_image()[:, :, ::-1])
    cv2.imwrite(os.path.join(sample_save_dir,d["file_name"]), out.get_image()[:, :, ::-1])
    cv2.imshow('inference sample',out.get_image()[:, :, ::-1])
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
embed()

### evaluate mAP ###
evaluator = COCOEvaluator("rail_test", ('bbox',), False, output_dir=sample_save_dir)
val_loader = build_detection_test_loader(cfg, "rail_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
