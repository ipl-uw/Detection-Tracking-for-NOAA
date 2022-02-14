import pandas as pd

# Change the path to the absolute path of the csv file on your PC.
csv_input = pd.read_csv('/home/jiemei/Documents/rail_detection/tracking_result_5_things/Predator_2018/20180824T194439-0800/tracking_result_with_huber.csv')

csv_input['length'] = 0
csv_input['location'] = 0
csv_input['depredation'] = 0
csv_input['release'] = 0
csv_input['injury'] = 0
csv_input['labeled_class'] = 0
csv_input['labeled_box'] = 0
csv_input['labeled_id'] = 0
csv_input['labeled_length'] = 0
csv_input['labeled_kept'] = 0
csv_input['comment'] = ""
csv_input['damages'] = ""
csv_input['occluded'] = 0
csv_input['comment2'] = ""
csv_input['comment3'] = ""
csv_input.head()

# The output is saved under the same folder. You can change the name.
csv_input.to_csv('/home/jiemei/Documents/rail_detection/tracking_result_5_things/Predator_2018/20180824T194439-0800/tracking_result_with_huber_processed.csv', index=False)