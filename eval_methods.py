import numpy as np
import more_itertools as mit
from sklearn.metrics import roc_curve, auc, precision_recall_curve, mean_squared_error, f1_score, precision_recall_fscore_support,confusion_matrix, precision_score, recall_score, roc_auc_score

def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                    A point is labeled as "anomaly" if its score is lower than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):
    Returns:
            np.ndarray: predict labels

    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    if label is None:
        predict = score > threshold
        return predict, 0

    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    for i in range(len(predict)):
        if any(actual[max(i, 0) : i + 1]) and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN,

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, adjust, topk=1):
    total_err_scores = total_err_scores.T
    normal_scores = normal_scores.T
    gt_labels = np.concatenate(gt_labels).tolist()
    total_features = total_err_scores.shape[0]
    # 取出最大的topK个scores值的索引值,得到每条样本中分数最大的特征索引值
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[
                   -topk:]
    total_topk_err_scores = []
    topk_err_score_map = []
    # 取出最大的topK个scores值
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    # 得到阈值
    thresold = np.max(normal_scores)

    if adjust:
        pred_labels, latency = adjust_predicts(total_topk_err_scores, np.array(gt_labels), thresold, calc_latency=True)
    else:
        # 根据标签得到最终预测的label
        pred_labels = np.zeros(len(total_topk_err_scores))
        pred_labels[total_topk_err_scores > thresold] = 1
        latency = 0

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    C = confusion_matrix(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return {
        "f1": f1,
        "precision": pre,
        "recall": rec,
        "TP": C[0, 0],
        "TN": C[1, 1],
        "FP": C[0, 1],
        "FN": C[1, 0],
        "threshold": thresold,
        "latency": latency,
        "roc_auc": auc_score,
    }

def maxval_eval(score, label, val_score, adjust=True):
    mean_ = np.mean(val_score)
    std_ = np.std(val_score)

    # val_score_scaler = preprocessing.MinMaxScaler().fit(val_score.reshape(-1, 1))
    # val_score = val_score_scaler.transform(val_score.reshape(-1, 1))
    # score = val_score_scaler.transform(score.reshape(-1, 1))

    val_max_threshold = mean_ + 3 * std_# np.max(val_score)

    if adjust:
        pred, p_latency = adjust_predicts(score, label, val_max_threshold, calc_latency=True)
    else:
        pred, p_latency = adjust_predicts(score, None, val_max_threshold, calc_latency=True)

    target, latency = calc_seq(score, label, val_max_threshold)

    fpr, tpr, ths = roc_curve(label, score)  # calculate fpr,tpr,ths
    roc_auc = auc(fpr, tpr)

    if label is not None:
        p_t = calc_point2point(pred, label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": val_max_threshold,
            "latency": p_latency,
            "roc_auc": roc_auc,
        }
    else:
        return {
            "threshold": val_max_threshold,
        }

def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True, adjust=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Finding best f1-score by searching for threshold..")
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target, latency = calc_seq(score, label, threshold, adjust)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_l = latency
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    fpr, tpr, ths = roc_curve(label, score)  # calculate fpr,tpr,ths
    roc_auc = auc(fpr, tpr)

    return {
        "f1": m[0],
        "precision": m[1],
        "recall": m[2],
        "TP": m[3],
        "TN": m[4],
        "FP": m[5],
        "FN": m[6],
        "threshold": m_t,
        "latency": m_l,
        "roc_auc":roc_auc,
    }


def calc_seq(score, label, threshold, adjust=True):
    if adjust:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    else:
        predict, latency = adjust_predicts(score, None, threshold, calc_latency=True)
    return calc_point2point(predict, label), latency