import numpy as np
import cv2
import json
import os
import re
from argparse import ArgumentParser

# score_threshold_dict =  {1: 0.5126, 2: 0.8644, 3: 0.1859, 4: 0.1859} # v2_test.json
score_threshold_dict =  {1: 0.5126, 2: 0.8644, 3: 0.8141, 4: 0.2110} # v2_train_overfit.json

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--box-result-json-path', type=str, default='', help='json path for predicted boxes from run_inference')
    parser.add_argument(
        '--data-dir', type=str, default='', help='path where the images are located')
    parser.add_argument(
        '--crops-save-dir', type=str, default='', help='path to directory where crops are saved')
    parser.add_argument(
        '--pred-COCO-JSON-path', type=str, default='', help='path to the predicted COCO JSON')
    args = parser.parse_args()
    return args

def extract_rotated_crop(img, box, img_name, count, crop_results_dir):
    '''
    https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
    '''
    cnt = np.array(box)
    # print("shape of cnt: {}".format(cnt.shape))
    rect = cv2.minAreaRect(cnt)
    # print("rect: {}".format(rect))

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # print("bounding box: {}".format(box))
    #cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))

    crop_name = img_name.split('.')[0] + '_crop_' + str(count) + '.png'
    cv2.imwrite(os.path.join(crop_results_dir, crop_name), warped)

def bbox2crop(args):
    '''
    Expected JSON format: {'name_frameID:[[[x_min, y_min], [x_max, y_max]], [[x_min, y_min], [x_max, y_max]], ......]'}
    '''
    crops = json.load(open(args.box_result_json_path))
    data_dir = args.data_dir
    crop_results_dir = args.crops_save_dir

    if not os.path.exists(crop_results_dir):
        os.mkdir(crop_results_dir)

    count = 0
    for img_name in os.listdir(data_dir):
        image = cv2.imread(data_dir + '/' + img_name)
        frame_id = re.findall(r'[A-Za-z]+|\d+', img_name)[1]
        for k, coords in crops.items():
            if k == img_name:
                for i in range(len(coords)):
                    x_min = coords[i][0][0]
                    y_min = coords[i][0][1]
                    x_max = coords[i][1][0]
                    y_max = coords[i][1][1]
                    cropped_image = image[y_min:y_max, x_min:x_max]
                    crop_name = 'frame_' + str(frame_id) + '_crop_'+str(i)+'.png' # add timestamp
                    cv2.imwrite(os.path.join(crop_results_dir, crop_name), cropped_image)
                count += 1

def crop_rbbox(args):
    coco_json = json.load(open(args.pred_COCO_JSON_path))
    data_dir = args.data_dir
    crop_results_dir = args.crops_save_dir

    if not os.path.exists(crop_results_dir):
        os.mkdir(crop_results_dir)

    count = 0
    for img_name in os.listdir(data_dir):
        image = cv2.imread(data_dir + '/' + img_name)
        detection_rboxes = coco_json[img_name.split('.')[0]]['detection_rboxes']
        detection_classes = coco_json[img_name.split('.')[0]]['detection_classes']
        detection_scores = coco_json[img_name.split('.')[0]]['detection_scores']
        for i, box in enumerate(detection_rboxes):
            # print(detection_scores[i], detection_classes[i], "thresh", score_threshold_dict[detection_classes[i]])
            if detection_scores[i] < score_threshold_dict[detection_classes[i]]: # eliminate all low confidence boxes
                # print("skipped ", count)
                continue
            extract_rotated_crop(image, box, img_name, count, crop_results_dir)
            count = count + 1
    
    print("Done!")

if __name__ == '__main__':
    args = parse_args()
    crop_rbbox(args)