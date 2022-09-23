from dataset_preprocess.train_test_split import rail_dataset_function
from dataset_preprocess.plt_dataset import plot_data_distribution
from IPython import embed
from detectron2.utils.visualizer import Visualizer

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
import random, cv2
from dataset_preprocess.xml_to_dict import category_ids

### Plot data distribution ###
data_path = './dataset_preprocess/rail_data/dataset_dicts.npz'
plot_data_distribution('./dataset_preprocess/rail_data/original_sp_count.npz',
                       './dataset_preprocess/rail_data/Alias_sp_count.npz',
                       './dataset_preprocess/rail_data/data distribution.png')


### register dataset ###
thing_classes = list(category_ids.keys())
for d in ["train", "test"]:
    DatasetCatalog.register("rail_" + d, lambda d=d:rail_dataset_function(data_path, d))
    MetadataCatalog.get("rail_" + d).set(thing_classes=thing_classes)
rail_metadata = MetadataCatalog.get("rail_train")


### show img label example ###
dataset_dicts = rail_dataset_function(data_path, 'train')

# for d in random.sample(dataset_dicts, 10):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=rail_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow('Train fish bbox example',out.get_image()[:, :, ::-1])
#     cv2.waitKey(3000)
#     cv2.destroyAllWindows()


### finetune a CoCo-pretrained R50-FPN Faster R-CNN ###
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.MODEL.DEVICE='cuda:1'
cfg.OUTPUT_DIR = './output_'+str(len(thing_classes))+'_things_sleeper_nonfish'
# cfg.OUTPUT_DIR = './output_'+str(len(thing_classes))+'_things'+'fintune_on_sleeper_shark_plus_chain_continue'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("rail_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = "/home/jiemei/Documents/rail_detection/output_5_things/model_final.pth"  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS = "/home/jiemei/Documents/rail_detection/output_5_thingsfintune_on_sleeper_shark_plus_chain/model_0199999.pth"  # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH= 3
cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
cfg.SOLVER.MAX_ITER = 500000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.CHECKPOINT_PERIOD=5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
# trainer.resume_or_load(resume=False)
trainer.train()

