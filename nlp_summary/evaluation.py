
import numpy as np
def precision_at_k(y_true_list, y_reco_list, users, k):
    precision_all = list()
    for u in users:
        y_true = y_true_list[u]
        y_reco = y_reco_list[u]
        common_items = set(y_reco).intersection(y_true)
        precision = len(common_items) / k
        precision_all.append(precision)
    return np.mean(precision_all)


def recall_at_k(y_true_list, y_reco_list, users, k):
    recall_all = list()
    for u in users:
        y_true = y_true_list[u]
        y_reco = y_reco_list[u]
        common_items = set(y_reco).intersection(y_true)
        if len(y_true) != 0:
            recall = len(common_items) / len(y_true)
        else:
            recall = 0
        recall_all.append(recall)
    return np.mean(recall_all)


def ndcg_at_k(y_true_list, y_reco_list, users, k):
    ndcg_all = list()
    for u in users:
        rank_list = np.zeros(k)
        y_true = list(set(y_true_list[u]))
        y_reco = y_reco_list[u]
        common_items, indices_in_true, indices_in_reco = np.intersect1d(
            y_true, y_reco, assume_unique=True, return_indices=True)

        if common_items.size > 0:
            rank_list[indices_in_reco] = 1
            ideal_list = np.sort(rank_list)[::-1]
            dcg = np.sum(rank_list / np.log2(np.arange(2, k + 2)))
            idcg = np.sum(ideal_list / np.log2(np.arange(2, k + 2)))
            ndcg = dcg / idcg
        else:
            ndcg = 0
        ndcg_all.append(ndcg)
    return np.mean(ndcg_all)

def evaluation(y_true_list, y_reco_list, users, k):
    precision_res = precision_at_k(y_true_list, y_reco_list, users, k)
    recall_res = recall_at_k(y_true_list, y_reco_list, users, k)
    ndcg_res = ndcg_at_k(y_true_list, y_reco_list, users, k)
    print("Precision@",k, ':', precision_res)
    print("Recall@:",k, ':', recall_res )
    print("nDCG@:", k, ':', ndcg_res)
    
    return precision_res, recall_res, ndcg_res 