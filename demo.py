import sys
import os
import shutil
import cv2
import numpy as np
from mmcv.image import imread, imwrite
from datetime import datetime
from django.conf import settings

def save_image(img_mem):
    img = imread(img_mem)
    basename = os.path.basename(img_mem)
    imgname = 'media/img/' + basename 
    copimg = 'media/ori_image/' + basename
    
    imwrite(img, imgname)
    imwrite(img, copimg)

    return imgname

def read_label_color(file, BGR=True):
    '''
        Parameters
        ----------
        file : txt file
            format: brownblight 255,102,0 orange
    
        Returns
        -------
        result : dict
            format: [ 'brownblight' : [255, 102, 0]]
    '''
    color_dict = {}
    
    with open(file, 'r') as f:
        lines = f.readlines()
    
    for l in lines:
        strs = l.split(' ')
        label = strs[0]
        color = list(map(int, strs[1].split(',')))
        if BGR:
            color[0], color[2] = color[2], color[0]
        color_dict[label] = color
        
    return color_dict

def pred_img(img_name):

    # print('import library')
    from fsdet.data.detection_utils import read_image
    from fsdet.config import get_cfg
    from predictor import VisualizationDemo

    cfg = get_cfg()
    cfg.merge_from_file('/home/eric/FSCE_tea-diseases/checkpoints/config.yaml')
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.freeze()

    demo = VisualizationDemo(cfg)
    
    img = read_image(img_name, format="BGR")
    predictions, visualized_output = demo.run_on_image(img)

    # bbox
    bboxes = np.array(predictions["instances"]._fields.get('pred_boxes').tensor.tolist())

    # scores
    scores = np.array(predictions["instances"]._fields.get('scores').tolist())

    # labels
    labels = np.array(predictions["instances"]._fields.get('pred_classes').tolist())

    # classes
    classes = ['brownblight', 'algal', 'blister', 'sunburn','fungi_early', 'roller',
            'moth', 'tortrix', 'flushworm', 'caloptilia', 'mosquito_early', 'mosquito_late',
            'miner', 'thrips', 'tetrany', 'formosa', 'other']

    # remove other
    if len(scores) != 0:
        # remove other
        bboxes, scores, labels = zip(*((x, y, z) for x, y, z in zip(bboxes, scores, labels) if z != 16)) # other's label is 16
        bboxes = np.array(list(bboxes))
        scores = np.array(list(scores))
        labels = np.array(list(labels))
       
    return labels, bboxes, classes, scores

def demo(img_name):

    labels, bboxes, classes, scores = pred_img(img_name)
    #print('drawing box')

    colorfile = os.path.join('color.txt')
    colors = read_label_color(colorfile)

    outfile = draw_det_bboxes_A(img_name,
                        bboxes,
                        labels,
                        colors=colors,
                        width=800,
                        class_names=classes,
                        score_thr=0.5,
                        out_file=img_name,
                        scores = scores)

    return outfile



def draw_det_bboxes_A(img_name,
                        bboxes,
                        labels,
                        colors,
                        width=None,
                        class_names=None,
                        score_thr=0.5,
                        out_file=None,
                        scores = None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        out_file (str or None): The filename to write the image.
    """
    if len(scores) != 0:
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    img = imread(img_name)
    img = img.copy()
    
    ori_size = img.shape

    ratio = width/ori_size[0]
    img = cv2.resize(img, (int(ori_size[1]*ratio),int(ori_size[0]*ratio)))
    
    ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    i = 0
    
    for bbox, label, score in zip(bboxes, labels, scores):
        
        pred_cls = class_names[label]
        color = colors[pred_cls]
        box_id = ABC[i]
        
        bbox = bbox*ratio
        bbox_int = bbox.astype(np.int32)
        
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        
        cv2.rectangle(img, (left_top[0], left_top[1]),
                      (right_bottom[0], right_bottom[1]), color, 4)
        text_size, baseline = cv2.getTextSize(box_id,
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)
        p1 = (left_top[0], left_top[1] + text_size[1])
        cv2.rectangle(img, tuple(left_top), (p1[0] + text_size[0], p1[1]+1 ), color, -1)
        cv2.putText(img, box_id, (p1[0], p1[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, 8)
        
        i += 1
        
        
    print('done   '+ str(out_file))
    
    if out_file is not None:
        imwrite(img, out_file)
    return out_file