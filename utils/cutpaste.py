from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import collections
import random
import shutil
import cv2
import os

from utils.general import xywh2xyxy, xyxy2xywh
from utils.datasets import img_formats, img2label_paths


# TODO add functions
'''
1. iter datasets(image already resized), extract all bounding boxes as single image
2. find vacant by target_shape
3. copy and paste
'''


def extract_boxes(path, img_size=640):
    '''
    resize image, then extract target boxes
    '''
    p = Path(path)  # images dir

    if p.is_file():
        files = []
        with open(p, 'r') as t:
            t = t.read().strip().splitlines()
            files += [Path(x).absolute() for x in t]  # local to global path

    elif p.is_dir():
        files = list(p.rglob('*.*'))
    else:
        raise Exception(f"the path for extract boxes is invalid : {p.as_posix()}")

    boxes_files_dict = collections.defaultdict(list)
    boxes_shapes_dict = collections.defaultdict(list)
    img_files = []
    
    outDir = files[0].parent.with_suffix('.classifier')
    shutil.rmtree(outDir) if outDir.is_dir() else None  # remove existing
    print(f"extract boxes to path: {str(outDir)}")
    
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
                img_files.append(im_file)
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
                    boxes_files_dict[c].append((f)) # append image name, and image shape
                    boxes_shapes_dict[c].append((b[3]-b[1], b[2]-b[0]))
                
            nf+=1
            pbar.set_description(f"extract boxes: found:{nf}, boxes:{nb}")

    assert len(boxes_files_dict)>0, f'no image found under : {path}'

    return boxes_files_dict, boxes_shapes_dict, img_files

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
    return inter_area / np.minimum(box1_area, box2_area)
    
def look_vacant(img, labels, hw):
    # image shape is (h, w, channel)
    h, w, c = img.shape
    assert c==3, 'channel order is not correct.'
    assert hw[0]<h or hw[1]<w, "vacant height and width must be less than image height and width"
    # retry 100 times
    for _ in range(100):
        # random generate left corner
        xmin,ymin = random.randint(0, w-hw[1]), random.randint(0, h-hw[0])
        box1 = np.array([xmin, ymin, xmin+hw[1], ymin+hw[0]]).reshape(-1,4).ravel().astype(np.float32)
        
        iou = bbox_iou(box1, labels[:,1:])
        if all(iou<0.2):
            return (xmin,ymin)

    return (None, None)


class CutPaste:

    def __init__(self, path, img_size):

        self.img_size = img_size

        boxes_files_dict, boxes_shapes, self.imgs = extract_boxes(path, self.img_size)
        self.class_nums = {}
        for c in boxes_files_dict:
            self.class_nums[c] = len(boxes_files_dict[c])

        # disable filter function
        # self.boxes_files_dict, self.boxes_shapes = filter_outliers(boxes_files_dict, boxes_shapes)
        self.boxes_files_dict, self.boxes_shapes = boxes_files_dict, boxes_shapes
        self.img_cache = collections.defaultdict(list)
        for c in self.boxes_files_dict:
            for img in self.boxes_files_dict[c]:
                self.img_cache[c].append(cv2.imread(str(img)))

        self.init_weights()
    
    def random_load_image(self, c):
        index = random.randint(0, len(self.boxes_files_dict[c])-1)
        if not self.img_cache:
            img = cv2.imread(str(self.boxes_files_dict[c][index]))  # BGR HWC
        else:
            img = self.img_cache[c][index]
        return img

    def init_weights(self):
        maxnum = max(self.class_nums.values())
        self.weights = ((maxnum-np.array(list(self.class_nums.values())))/len(self.imgs)).tolist()
        self.weights = [w/max(self.weights) for w in self.weights]
        print("cut-paste weight: %s" % self.weights)
    def cut_paste(self, img, labels):
        '''
        img: HWC BGR
        '''
        assert img.shape[2] == 3, "channel order is not corrects"
        assert img.shape == (self.img_size, self.img_size, 3), \
                            f"image shape does not matched: {img.shape}, exp:{(self.img_size, self.img_size, 3)}"
        assert labels.shape[1] == 5, 'labels format does not matched'
        
        newimg = img.copy()
        newlabels = labels.copy()

        for c in [c for c in self.class_nums if self.boxes_files_dict[c] and self.weights[c]>0]:
            prob = self.weights[c]
            while prob>0:
                if prob<1 and random.random()>prob:
                    break
                prob-=1
                # get image for this target
                c_img = self.random_load_image(c)
                c_hw = c_img.shape[:2]
                # get the center for this target
                x, y = look_vacant(img, newlabels, c_hw)
                if x is not None and y is not None:
                    newimg[y:y+c_hw[0], x:x+c_hw[1]] = c_img
                    label = np.array([c,x,y,x+c_hw[1],y+c_hw[0]],dtype=np.float32, ndmin=2) # class, xyxy
                    newlabels = np.concatenate((newlabels, label), axis=0)

        # Visualize
        flag = False
        # flag = True
        if flag:
            import matplotlib.pyplot as plt
            ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
            ax[0].imshow(img[:, :, ::-1]), ax[0].set_title('original image')
            ax[1].imshow(newimg[:, :, ::-1]), ax[1].set_title('new image')
            plt.savefig('compare_cutpaste_before_after.jpg')
            # for save new label and new image
            new = newlabels.copy()
            h,w,c = newimg.shape
            new[:,1:] = xyxy2xywh(new[:,1:])
            new = new.T
            new[1],new[3] = new[1]/w, new[3]/w
            new[2],new[4] = new[2]/h, new[4]/h
            np.savetxt("cut_paste.txt", new.T, '%10.3g')
            im = Image.fromarray(newimg)
            im.save("cut_paste.jpg")
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