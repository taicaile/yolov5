# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs=None):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # tcls 包含每个输出层检测出来的类别
        # tbox 包含每个输出层检测出来的目标的 grid的偏移量（已经减去grid index）
        # indices 包含 image id, anchor id, grid indices(整数), is index of tbox
        # anchors 包含每个输出层检测出来的目标 对应匹配的anchor
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # tcls[0].shape: [484]
        # tbox[0].shape: [484,4]
        # anchors[0].shape: [484,2]

        # DEBUG
        # plot anchors
        # if imgs is not None:
        #   from utils.iplot import plot_build_targets
        #   plot_build_targets(imgs, targets, model, tcls, tbox, indices)
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                # 选择出target存在的prediction subset
                # pi 是predictions的当前layer的输出
                # pi[0].shape : [16, 3, 80, 80, 85]
                # 16 个图片，每一个layer有三个anchor，第一层grid是80×80(img_size=640)
                # image index, anchor index, grid y, grid x
                # 注意，这里已经是筛选 grid x index和 grid y index 之后的target
                '''
                ISSUE, 每个layer的每个cell最多可以预测5个target, 每个target最多有三个anchor, 也就是每个cell最多15个targets.
                但是预测网络的输出是每个layer的每个cell最多可以输出三个目标。此时就会有重复利用的问题。
                b, a, gj, gi 是build targets 返回的index， 可能包含重复的index。
                下面是在预测的网格中取出包含targets的预测值， ps为仅包含build targets产生的targets。
                ps = pi[b, a, gj, gi]
                ps也可能包含重复的预测，此时的预测和真正的target的box并不匹配。
                同一个 b,a,gj,gi 可以代表不同的目标，但是ps仅包含唯一的目标。
                '''
                '''
                (Pdb) p b.shape
                torch.Size([303])
                unique, counts = np.unique(np.stack((b,a,gj,gi), axis=0).T, return_counts=True, axis=0)
                (Pdb) unique.shape
                (300, 4)
                说明包含重复的index
                '''
                # unique, counts = np.unique(ps.detach().clone().numpy(), return_counts=True, axis=0)
                # sum(counts>1)=3, the result greater than 1 saying use same prediction corresponding multiple targets.
                # ps.shape : [484, 85] 
                # Regression
                # sigmoid range [0,1], pxy range [-0.5,1.5], 0.5是网格的中心，－0.5和1.5代表临近网格的中心。
                # 计算检测头输出的x,y, 这里已经不包含grid index， x和y分别是对应当前grid网格内的偏移

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # wh 是 anchor 的0－4倍
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                # 预测层输出的，对应grid网格内的x和y，以及w和h
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # tbox 是 targets 对应的box，包含当前网格内的偏移量 x,y, 以及对应anchor的比例 w,h
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # iou loss 取(1-iou)中值？
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # target objectness score, 是否有目标存在的评分, 用实际和预测的 iou 作为target
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # 分类loss
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # ps.shape : [484, 85]
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    # t.shape : [484, 80], 80 classes
                    # class positive, 得到分类target
                    t[range(n), tcls[i]] = self.cp
                    # t.sum().sum() = 484, 计算BCE LOSS
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            
            # 计算obj loss with BCE, balance 为不同layer的weights
            # 有目标存在的，其目标objectness是iou, 否则目标为0
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # targets 包含6列，第0列为 batch 中的image id. 然后为 class id, x,y,w,h
        # targets.shape : [142,6]
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # 获得检测头模块
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # 创建二维数组，对应三个anchor， 142个label
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # ai.shape:[3,142]
        # repeat targets 和 anchors 一致, 然后追加 anchor index 作为最后一列
        # (targets.repeat(na, 1, 1).shape : [3, 142, 6]
        # ai[:, :, None].shape:[3, 142, 1]
        # before : targets.shape: [142,6]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        # after, targets.shape:[3, 142, 7], 3 layers
        g = 0.5  # bias # bias 网格中心偏移
        # 附近四个网格，加上自身
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        '''
        tensor([[ 0.00000,  0.00000],
                [ 0.50000,  0.00000],
                [ 0.00000,  0.50000],
                [-0.50000,  0.00000],
                [ 0.00000, -0.50000]])
        '''
        # nl 为检测头输出分支
        for i in range(self.nl):
            # det.anchors.shape : [3,3,2]
            # anchors.shape : [3,2]
            anchors = self.anchors[i]
            # p 为检测头输出，3个分支 :
            # p[0] shape: 16*3*80*80*85, 
            # p[1] 16*3*40*40*85, 
            # p[2] 16*3*20*20*85
            # targets 的增益, 将 targets的 x和y，映射到grid网格上
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # gain : [ 1.,  1., 80., 80., 80., 80.,  1.]
            # gain : img id, class id, x,y,w,h, anchor index.
            # Match targets to anchors
            # t 为影射grid网格index上的target
            # Match targets to anchors
            t = targets * gain
            # t.shape : [3, 142, 7]
            # nt 为 target 的数量
            if nt:
                # Matches
                # t[:, :, 4:6].shape : [3, 142, 2]
                # anchors[:, None].shape : [3, 1, 2]
                # 预测出来的box和预设的anchor，w和h的缩放系数
                # 注意这里的anchors的数值,对应的是grid cell的倍数, 而不是图片像素。
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                # r.shape : [3,142,2], 每个grid预测3个目标，一共包含142个目标，w 和 h
                # model.hyp['anchor_t']=4.0
                # 根据不同layer的不同anchors参数，来筛选出w和h比例不超过4倍的target。
                # 同一个target或许可以匹配2个layer的不同anchors，具体要看参数的配置。
                # 同一个target可能被同一个layer两个anchor同时匹配上，具体看参数配置。
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j.shape:[3, 142]
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # before : t.shape : [3,142,7]
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter
                # after : t.shape : [162, 7]
                # Offsets
                # target 映射到grid 网格的x,y
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                # https://www.kaggle.com/c/global-wheat-detection/discussion/172436
                # j,k,l,m 是临近的4个邻居，在其中选择两个，加上自身就是三个
                # 每一个layer, 每个cell(不包含边缘cell)可以预测9个objects, 
                # 每个cell 3个anchors 以及每个target可以包含三个cell, 3*3=9
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # j.shape: [5,162], 5 5个之中有三个是True
                # t.shape: [162,7]
                # t.repeat((5,1,1)).shape:[5, 162, 7]
                t = t.repeat((5, 1, 1))[j]
                # t.shape:[484, 7]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                # offsets.shape : [484, 2]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image id, class id
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
