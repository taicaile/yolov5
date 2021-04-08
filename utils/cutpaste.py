from pathlib import Path
from tqdm import tqdm

import numpy as np
import collections
import random
import shutil
import cv2
import os

from utils.general import xywh2xyxy
from utils.datasets import img_formats, img2label_paths


# TODO add functions
'''
1. iter datasets(image already resized), extract all bounding boxes as single image
2. find vacant by target_shape
3. copy and paste
'''


def extract_boxes(path, img_size=640):
    imgFiles = collections.defaultdict(list)
    shapes = collections.defaultdict(list)

    path = Path(path)  # images dir
    outDir = path.parent.with_suffix('.classifier')
    shutil.rmtree(outDir) if outDir.is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    pbar = tqdm(files, total=n)
    nf = nb = 0
    for im_file in pbar:
        if im_file.suffix[1:] in img_formats:
            # image
            im = cv2.imread(str(im_file)) # [..., ::-1]  # BGR to RGB WHY?

            h0, w0 = im.shape[:2]  # orig hw
            r = img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_LINEAR
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            h, w = im.shape[:2] # resize hw

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = outDir / f'{c}' / f'{im_file.stem}_{c}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    # UPDATE cancel pad
                    # b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'
                    nb+=1
                    imgFiles[c].append((f)) # append image name, and image shape
                    shapes[c].append((b[3]-b[1], b[2]-b[0]))
            nf+=1
            pbar.set_description(f"extract boxes: found:{nf}, boxes:{nb}")

    assert len(imgFiles)>0, f'no image found under : {path}'

    return outDir, imgFiles, shapes

def bbox_iou(box1, box2):
    # Returns the min intersection over box1 or box2 area given box1, box2. 
    # box1 is 4, box2 is nx4. boxes are x1y1x2y2
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                    (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box1 area
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16
    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # return max intersection over box1 or box2 area
    return inter_area / min(box1_area, box2_area)
    
def look_vacant(image, labels, hw):
    # image shape is (h, w, channel)
    h, w, c = image.shape
    assert c==3, 'channel order is not correct.'

    retry = 0
    # retry 10 times
    while retry <= 100:
        # random generate left corner
        if w-hw[1]<=0 or h-hw[0]<=0:
            break
        xmin = max(0, random.randint(0, w-hw[1]))
        ymin = max(0, random.randint(0, h-hw[0]))
        # box1 int
        box1 = np.array([xmin, ymin, hw[1], hw[0]]).reshape(-1,4).ravel().astype(np.float32)
        # convert box1 to float
        box1[[0, 2]]/=w  # normalized width 0-1
        box1[[1, 3]]/=h  # normalized height 0-1
        # box1[0],box1[2] =  box1[0]/w, box1[1]/w
        # box1[1],box1[3] =  box1[1]/h, box1[3]/h

        for box in xywh2xyxy(labels[:, -4:]):
            # box is float
            # calc iou, iou needs less than 0.2
            iou = bbox_iou(box1, box)
            if iou > 0.2:
                break
        else:
            return (xmin,ymin)

        retry+=1

    # print("no vacant found")
    return (None, None)


class CutPaste:
    # TODO 
    '''
    scan custom-vehicles only
    1, mannuly scan.
    2, rebuid function, need to resize first when extract boxes.
    3, how to check which object is full or part of it.
    4, paste object
    '''
    def __init__(self, path, img_size=640, classes=[]):
        assert os.path.exists(path), f"The path:{path} for cut-paste is not valid!!!"
        self.img_size = img_size
        self.img_path, img_files, shapes = extract_boxes(path, img_size)
        self.img_files, self.shapes = filter_outliers(img_files, shapes)
        self.classes = classes
        self.update_weights()
    
    def load_image(self, c):
        index = random.randint(0, len(self.img_files[c])-1)
        img = cv2.imread(str(self.img_files[c][index]))  # BGR
        # img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        return img

    def update_weights(self):
        # banlance object by mAPs of each objects
        # then decide how many objects needed to add this image
        # update self.classes
        pass

    def cut_paste(self, img, labels):
        if not self.classes:
            return img, labels

        assert img.shape[2] == 3, "channel order is not corrects"
        assert img.shape == (self.img_size, self.img_size, 3), f"image shape does not matched \
                                            : {img.shape}, exp:{(self.img_size, self.img_size, 3)}"
        assert labels.shape[1] == 5, 'labels format does not matched'
        
        newimg = img.copy()
        newlabels = labels.copy()
        for c in [c for c in self.classes if self.img_files[c]]:
            # get image for this target
            c_img = self.load_image(c)
            c_hw = c_img.shape[:2]
            # get the center for this target
            x, y = look_vacant(img, newlabels, c_hw)
            if x is not None and y is not None:
                newimg[y:y+c_hw[0], x:x+c_hw[1]] = c_img
                label = np.array([c,x,y,x+c_hw[1],y+c_hw[0]],dtype=np.float32, ndmin=2) # class, xyxy
                newlabels = np.concatenate((newlabels, label), axis=0)

        # Visualize
        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
        # ax[0].imshow(img[:, :, ::-1])  # base
        # ax[0].set_title('original image')
        # ax[1].imshow(newimg[:, :, ::-1])  # warped
        # ax[1].set_title('new image')
        # plt.savefig('compare.jpg')

        return newimg, newlabels

def filter_outliers(img_files, shapes):
    img_files_dict = collections.defaultdict(list)
    shapes_dict = collections.defaultdict(list)
    
    for c, shapes in shapes.items():
        assert len(shapes), f'no images found on category : {c}'
        ratios = np.array([item[0]/item[1] for item in shapes])
        hist, bin_edges = np.histogram(ratios, bins='auto')
        index = np.digitize(ratios, bin_edges) == (hist.argmax() + 1)
        img_files_dict[c] = np.array(img_files[c])[index].tolist()
        shapes_dict[c] = np.array(shapes)[index].tolist()

    assert len(img_files_dict) == len(shapes_dict), f'img_files and shapes length not same, \
                                                        {len(img_files_dict)} != {len(shapes_dict)}'
    return img_files_dict, shapes_dict

def extract_boxes_from_dataset(dataset):

    # Convert detection dataset into classification dataset, with one directory per class
    classifier_path = Path(dataset.img_files[0]).parent.parent/'classifier' # cached labels
    shutil.rmtree(classifier_path) if classifier_path.is_dir() else None  # remove existing

    n = len(dataset)  # number of files
    pbar = tqdm(dataset, total=n, desc=f"generate boxes on {classifier_path}")
    mosaic = dataset.mosaic
    augment = dataset.augment
    
    dataset.mosaic = False
    dataset.augment = False

    # torch.from_numpy(img), labels_out, self.img_files[index], shapes
    for img, label, img_file, shape in pbar:
        img = img.numpy().transpose(1, 2, 0)  # BGR to RGB
        label = label.numpy()
        # image
        h = dataset.img_size
        w = dataset.img_size

        lb = label[:,1:]  # labels

        for j, x in enumerate(lb):
            c = int(x[0])  # class
            f = classifier_path / f'{c}' / f'{classifier_path.stem}_{Path(img_file).stem}_{j}.jpg'  # new filename
            if not f.parent.is_dir():
                f.parent.mkdir(parents=True)

            b = x[1:] * [w, h, w, h]  # box
            # b[2:] = b[2:].max()  # rectangle to square
            # UPDATE cancel pad
            # b[2:] = b[2:] * 1.2 + 3  # pad
            b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

            b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
            b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
            assert cv2.imwrite(str(f), img[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'

    dataset.mosaic = mosaic
    dataset.augment = augment