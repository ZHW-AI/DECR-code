import numpy as np
from skimage import measure
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def metric_cal(scores, gt_list, gt_mask_list, cal_pro=True):#(149, 224, 224)-->(n,h,w)
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)#通过重塑和计算最大值，可以获得每个样本的最高得分
    gt_list = np.asarray(gt_list, dtype=int)#列表 gt_list 转换为一个 NumPy 数组，并指定了数据类型为整数（int）
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    # print('INFO: image ROCAUC: %.3f' % (img_roc_auc))

    # get optimal threshold
    gt_mask = np.asarray(gt_mask_list, dtype=int)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    # calculate pro
    if cal_pro:
        pro_auc_score = cal_pro_metric(gt_mask_list, scores, fpr_thresh=0.3)
    else:
        pro_auc_score = 0

    return img_roc_auc, per_pixel_rocauc, pro_auc_score, threshold

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def cal_pro_metric(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)#将labeled_imgs列表转成数组
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.bool)
# 以上这一步只是为了更加确保标签图像是二值化的，因为本身labeled_imgs本身就是二值图；
    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps
# 以上通过计算最大值和最小值，得到预测分数的范围，然后将范围分成max_steps个部分，每个部分的宽度即为步长。
    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool)
# 初始化为全零数组的目的是为了确保在后续的操作中，所有像素的初始值都是相同的，并且都是False（0）。这样可以在二值化过程中，只需要关注预测得分是否大于阈值，而不用考虑初始值的影响。
    for step in range(max_steps): #max_steps=200
        thred = max_th - step * delta#计算阈值用于阈值分割
        # segmentation--根据阈值 thred 将 score_imgs 中的得分转换为二值的分割结果
        binary_score_maps[score_imgs <= thred] = 0#(149, 224, 224)数组
        binary_score_maps[score_imgs > thred] = 1#(149, 224, 224)数组

        pro = []  # per region overlap-用于存储每个区域的重叠像素数
        iou = []  # per image iou-用于存储每个图像的 IoU 值
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            # 目的：找到二值图labeled_imgs[i]中的每个连通区域，然后计算每个连通区域的一些属性，如面积、边界框等。
            
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                # 对于每个连通区域，通过prop.bbox获取其边界框的坐标(x_min, y_min, x_max, y_max)。
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # 从binary_score_maps[i]中提取连通区域对应的预测结果，并存储在cropped_pred_label中。
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                # 从prop.filled_image中提取连通区域对应的真实标签，并存储在cropped_mask中。prop.filled_image表示连通区域prop对应的二值化真实标签。连通区域是指图像中相邻的像素具有相同标签（像素值）的像素集合。
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                # 计算cropped_pred_label和cropped_mask的交集，并将结果转换为浮点型后求和，得到‘交集’的像素个数。
                pro.append(intersection / prop.area)
                # 将交集的像素个数除以连通区域的面积(prop.area)，得到预测结果与真实标签的交集区域占连通区域的比例。
            
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            # intersection表示预测分割结果binary_score_maps[i]和真实标签图像labeled_imgs[i]的交集中的像素数（即重叠的像素数）
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            # union表示预测分割结果binary_score_maps[i]和真实标签图像labeled_imgs[i]的并集中的像素数。
            
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
            # labeled_imgs[i].any()用于判断真实标签图像labeled_imgs[i]是否存在异常像素（即是否存在标签值为1的像素）。
            # 如果真实标签图像中没有异常像素，即所有像素值都为0，那么计算IoU是没有意义的，因为没有预测结果会与真实标签图像重叠。因此，当真实标签图像没有异常像素时，跳过计算IoU的步骤，不将其添加到iou列表中。
        
        ###### 以上for循环，通过像素级的IoU计算来评估模型预测的分割效果。
        ###### 对于每个图像，计算预测分割结果与真实标签的交集和并集，并计算IoU值。
        ###### 对于每个区域，计算其与预测分割结果的重叠像素数，并计算区域级的IoU值。
        
        # 计算每个图像的平均IoU值和标准差。
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        # print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        
        # 计算每个连通区域的平均IoU值和标准差。
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())

        # fpr for pro-auc--得到了不同阈值下的假阳率和对应的阈值列表，用于后续计算P-R曲线和AUC值等指标。
        masks_neg = ~labeled_imgs
        # ~labeled_imgs操作会将gt_mask中的非零像素值变为0，零像素值变为1，从而得到非真实标签的掩膜。
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        # 计算非真实标签掩膜和预测结果二值化的交集（即两者都为1的位置），再通过masks_neg.sum()计算非真实标签掩膜中1的总数，从而得到假阳率（False Positive Rate，FPR）。
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc--目的是计算模型在指定假阳率阈值下的P-R曲线下的面积
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    # 通过rescale函数将假阳率fprs_selected进行重新缩放，将其范围从[0, 0.3]缩放到[0, 1]，这是为了计算AUC值时的输入范围要求。
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # 使用auc函数计算在经过筛选后的假阳率和pros_mean值组成的曲线下的面积，得到pro_auc_score，即P-R曲线下的面积，
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score