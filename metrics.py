import numpy as np
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
"""
confusionMetric  # 注意：此处竖着代表预测值，横着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass+1
        self.confusionMatrix = np.zeros((self.numClass,) * 2)
        #print(self.confusionMatrix.shape)
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
    def computeRecall(self):# sum(axis=1)表示按照行加（每一行加起来）   召回率： 正样本中，预测正确的比例
        # recall1 = TP/TP+FN          recall2 = TN/TN+FP
        smooth = 1e-5
        #recall = np.diag(self.confusionMatrix)/np.sum(self.confusionMatrix,axis=1) + smooth
        recall = np.diag(self.confusionMatrix)/self.confusionMatrix.sum(axis=0) + smooth
        #print('=====================recall', recall)
        return recall[0]
    def computeSpec(self): # 负样本中预测正确的比例
        # Spec = TN/(FP+TN)
        smooth = 1e-5
        recall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0) + smooth
        return recall[1]

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        smooth = 1e-5
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1) + smooth
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        
        #print(type(label))
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        if torch.is_tensor(imgPredict):
            imgPredict = torch.sigmoid(imgPredict).data.cpu().numpy()
        if torch.is_tensor(imgLabel):
            imgLabel = imgLabel.data.cpu().numpy()
        imgPredict_ = imgPredict > 0.5
        imgLabel_ = imgLabel > 0.5
        self.confusionMatrix = self.genConfusionMatrix(imgPredict_, imgLabel_)
        #print('2==================\n',self.confusionMatrix)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)  # F1 score
    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

class NT_Xent(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, feat1, feat2):
        """
        implement based on pos_mask & neg_mask; could also use torch.diag & nn.CrossEntropyLoss
        Args:
            feat1, feat2: feats of view1, view2; feat1.shape == feat2.shape == (batch_size, C)
        Returns:
            A loss scalar.
        """
        # works for DataParallel; default cuda:0
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        batch_size = feat1.shape[0]
        # compute logits
        features = torch.cat([feat1, feat2], dim=0)
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # neg_mask: denominator; mask out self-contrast cases
        neg_mask = ~torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        '''neg_mask, batch_size=4
                    |
            0 1 1 1 | 1 1 1 1
            1 0 1 1 | 1 1 1 1
            1 1 0 1 | 1 1 1 1
            1 1 1 0 | 1 1 1 1
          ----------|---------- 
            1 1 1 1 | 0 1 1 1
            1 1 1 1 | 1 0 1 1
            1 1 1 1 | 1 1 0 1
            1 1 1 1 | 1 1 1 0
                    |
        '''
        # pos_mask: numerator; single positive pair
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool).to(device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)
        '''pos_mask, batch_size=4
                    |
            0 0 0 0 | 1 0 0 0
            0 0 0 0 | 0 1 0 0
            0 0 0 0 | 0 0 1 0
            0 0 0 0 | 0 0 0 1
          ----------|---------- 
            1 0 0 0 | 0 0 0 0
            0 1 0 0 | 0 0 0 0
            0 0 1 0 | 0 0 0 0
            0 0 0 1 | 0 0 0 0
                    |
        '''

        # compute log_prob
        exp_logits = torch.exp(logits)[neg_mask].view(2 * batch_size, -1)
        log_prob = logits[pos_mask] - torch.log(exp_logits.sum(1))

        # loss
        loss = -(self.temperature / self.base_temperature) * log_prob.mean()
        return loss
