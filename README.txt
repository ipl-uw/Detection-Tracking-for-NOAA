Train Faster RCNN, with all historical data.

1. 数据生成
check_img_size.py: 删除损坏的图像和label。
xml_to_dict.py: 仿照detectron2 tutorial (https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0) 给的 coco数据格式，写的一个子函数。这里有处理Alias name。输入是一个xml文件。
rail_dataset.py: 把所有label的xml文件，转换成coco格式，存成dataset_dicts.npz格式，因为有shuffle，前90%是train， 后10%是test，直接给到train.py。同时保存 找到alias之后的完整的data 和 原始name的统计Alias_sp_count.npz，original_sp_count.npz。遍历所有xml文件，利用xml_to_dict.py 保存成dict的数据格式。class_alias.py里是需要的label名字，如果需要增加或者修改名字，在这里修改。

train_test_split.py: 用于train.py直接读取label的npz。
																																																																																													plt_dataset.py:画出数据分布。

2. train.py:
仿照detectron2 tutorial (https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0) 写的code。weights 保存在output文件夹里。

3. evaluation.py: 在test data上测试 AP. 并存下几个sample可视化。存在output_eva 文件夹里。

4. inference_for_rail.py: 在rail unlabeled的data上测试，把结果存成csv文件，包含没有detection bbox的frame。在detection_result文件夹里。

5. detection_demo.py: generate video based on csv detection file


Tracking,iou-tracker 文件夹里
1. demo.py: 进行visual tracking， 只会把有tracking 的frame保存在csv里面。会调用set_roi设置 detection bbox的ROI, 同时保存det_ROI.npy。
2. tracking_demo.py：会对已有的tracklets，进行huber regression， 并把所有frame 包括没有detectionframe，写成video进行可视化。data_huber之后的tracking结果，保存到tracking_result_with_huber.csv，以便Kelsey之后进行人工检查。
3. rail_detection/Preprocess csv.py 修改tracking_result_with_huber.csv 里的header,保留几个我的算法的输出，但是增加GUI需要的其他项目。




