
import cv2
from numpy.core.defchararray import array, index
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from utils.general import xywh2xyxy
from utils.plots import color_list, plot_one_box
from utils.torch_utils import is_parallel

def tensor2numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    return tensor

# plot grid
def plot_grid(img, stride):
    assert img.shape[2] == 3, 'input img channel order is not as expected'
    hw = np.array(img.shape[:2])
    assert (hw%stride==0).all(), "height or width is not times of stride, hw: %s" % hw
    xs = np.arange(start=stride, stop=hw[0], step=stride)
    ys = np.arange(start=stride, stop=hw[1], step=stride)
    
    for x in xs:
        cv2.line(img, (int(x), 0), (int(x), hw[1]), (192, 192, 192),1 )
    for y in ys:
        cv2.line(img, (0, int(y)), (hw[0], int(y)), (192, 192, 192),1 )

def plot_targets(img, targets, colors, names=None):
    h, w = img.shape[:2]
    boxes = xywh2xyxy(targets[:, 2:6]).T
    classes = targets[:, 1].astype('int')
    labels = targets.shape[1] == 6  # labels if no conf column
    conf = None if labels else targets[:, 6]  # check for confidence presence (label vs pred)

    if boxes.shape[1]:
        if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
            boxes[[0, 2]] *= w  # scale to pixels
            boxes[[1, 3]] *= h
    for j, box in enumerate(boxes.T):
        cls = int(classes[j])
        color = colors[cls % len(colors)]
        cls = names[cls] if names else cls
        if labels or conf[j] > 0.25:  # 0.25 conf thresh
            label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
            plot_one_box(box, img, label=label, color=color)

def plot_anchors(img, anchors, stride):
    assert img.shape[2] == 3, "img channel order error, expect img.shape[2]=3"
    # 从 grid cell 还原到像素
    anchorspix = anchors * stride# (np.array(img.shape[:2])/stride)
    center = np.array(img.shape[:2])//2
    for i, anchor in enumerate(anchorspix):
        label = '[%s]' % (i)
        box = xywh2xyxy(np.concatenate((center, anchor)).reshape((1,4)))[0]
        plot_one_box(box, img, label=label)

def plot_build_targets(imgs, targets, model, tcls, tbox, indices,
                        fname=None, max_subplots=5, classes=None):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    # get stride from model.stride
    stride = det.stride.cpu().numpy().astype(np.int)
    anchors = det.anchors.cpu().numpy()

    # Filter targets by classes
    if classes is not None:
        targets = targets[(targets[:, 1] == np.array(classes)).any(1)]

    # un-normalise
    if np.max(imgs[0]) <= 1:
        imgs *= 255
    ni, _, h, w = imgs.shape  # batch size, _, height, width
    ni = min(ni, max_subplots)  # limit plot images
    nc = len(stride)
    colors = color_list()  # list of colors
    mosaic = np.full((int(ni * h), int(nc * w), 3), 255, dtype=np.uint8)  # init

    for i, img in enumerate(imgs[:ni]):
        block_y = int(h * i)
        img = img.transpose(1, 2, 0) # back to (h,w,channel)
        # 3 layers
        for j in range(nc):
            image = np.full((h, w, 3), 255, dtype=np.uint8)  # init
            image[:] = img[:]

            jtcls = tcls[j].cpu().numpy()
            jtbox = tbox[j].cpu().numpy()

            b, a, gj, gi = indices[j]  # image index, anchor index, gridy, gridx
            b = b.cpu().numpy()
            a = a.cpu().numpy()
            gj = gj.cpu().numpy()
            gi = gi.cpu().numpy()
            # filter by image id
            ifilter=b==i
            b = b[ifilter]
            a = a[ifilter]
            gj = gj[ifilter]
            gi = gi[ifilter]
            jtcls = jtcls[ifilter]
            jtbox = jtbox[ifilter]

            # filter by classes
            if classes is not None:
                cfilter = (jtcls.reshape(-1, 1) == np.array(classes)).any(1)
                jtcls = jtcls[cfilter]
                jtbox = jtbox[cfilter]
                b = b[cfilter]
                a = a[cfilter]
                gj = gj[cfilter]
                gi = gi[cfilter]

            gij = np.concatenate((gi.reshape(-1,1), gj.reshape(-1,1)), axis=1)
            unique, counts = np.unique(gij, return_counts=True, axis=0)
            for index, ij in enumerate(unique):
                c1 = tuple((ij*stride[j].astype(np.int)).tolist())
                c2 = tuple(((ij+1)*stride[j].astype(np.int)).tolist())
                color = (255, 255, 0)
                cv2.rectangle(image, c1, c2, color=color, thickness=-1, lineType=cv2.LINE_AA)  # filled
                label = str(counts[index])
                center = [c1[0], c2[1]]
                tl = max(stride[j]//8, 1)
                cv2.putText(image, label, tuple(center), fontFace=0, fontScale=tl/3, color=[128, 0, 128], 
                            thickness=tl, lineType=cv2.LINE_AA)

            # imgfilter=b==np.array([i])
            block_x = int(w * j)
            # plot grid
            plot_grid(image, stride[j])
            # plot anchors
            plot_anchors(image, anchors[j], stride[j])
            # plot targets
            if len(targets) > 0:
                image_targets = targets[targets[:, 0] == i]
                plot_targets(image, image_targets, colors)
            # plot build targets
            
            # Image border
            mosaic[block_y:block_y + h, block_x:block_x + w, :] = image
            cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    fname = 'loss.jpg'
    Image.fromarray(mosaic).save(fname)  # PIL save