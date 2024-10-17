import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve
import matplotlib.pyplot as plt

def fmax(probs, labels):
    thresholds = np.arange(0, 1, 0.01)
    f_max = 0.0

    for threshold in thresholds:
        precision = 0.0
        recall = 0.0
        precision_cnt = 0
        recall_cnt = 0
        for idx in range(probs.shape[0]):
            prob = probs[idx]
            label = labels[idx]
            pred = (prob > threshold).astype(np.int32)
            correct_sum = np.sum(label*pred)
            pred_sum = np.sum(pred)
            label_sum = np.sum(label)
            if pred_sum > 0:
                precision += correct_sum/pred_sum
                precision_cnt += 1
            if label_sum > 0:
                recall += correct_sum/label_sum
            recall_cnt += 1
        if recall_cnt > 0:
            recall = recall / recall_cnt
        else:
            recall = 0
        if precision_cnt > 0:
            precision = precision / precision_cnt
        else:
            precision = 0
        f = (2.*precision*recall)/max(precision+recall, 1e-8)
        f_max = max(f, f_max)

    return f_max


def auprc(probs, labels):
    """
    Area under precision-recall curve (AUPRC).

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    return average_precision_score(labels, probs, average='micro')


def plot_prc(probs, labels): # 严格定义计算方法
    precision, recall, thresholds = precision_recall_curve(labels, probs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.grid()  # 生成网格
    plt.plot(recall,precision)
    plt.figure("P-R Curve")
    plt.show()

    return auc(recall, precision)


