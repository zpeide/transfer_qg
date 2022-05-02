import torch
import time
from transformers import *
import numpy as np
import os, json

from collections import defaultdict
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS
from sklearn import cluster, mixture
# t-sne
from sklearn import manifold
import sklearn
from sklearn.metrics import confusion_matrix
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import random

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.spatial import distance


random.seed(21)
np.random.seed(21)


colors = ['#F15BB5', '#FEE440', '#00BBF9', '#02C39A', '#FDC500', '#00509D', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'black']
markers = ['o', '*', '^']
base_path_new = '../../data'

file_paths_new = {
                'NQ': os.path.join(base_path_new, 'nq/train.jsonl'),
                'RACE': os.path.join(base_path_new, 'RACE/dev.jsonl'),
                'SciQ':  os.path.join(base_path_new, 'SciQ/test.jsonl'),
            }
models_to_use = ['bert-base-uncased']




def scatter_unsupervised_results(data, labels_):
    labels = set(labels_)
    print(len(labels))
    colors = [plt.cm.tab20(each) for each in np.linspace(0, 1, 10)] #len(labels))]
    # colors = [plt.cm.Spectral(each)
    #         for each in np.linspace(0, 1, len(labels))]
    # colors = ['green', 'b', 'orange']
    for _j, l in enumerate(list(labels)[:]):
        items = np.concatenate( [np.expand_dims(data[i, :], axis=0) for i in range(data.shape[0]) if labels_[i] == l] )
        print(items.shape)
        plt.scatter(items[:, 0], items[:, 1], alpha=0.3, color=colors[_j], label=str(l))
    if len(labels) < 5:
        plt.legend()


def assign_cluster_label_with_type(true_labels, labels_):
    clusters = list(set(labels_))
    true_types = list(set(true_labels))

    cluster_items = {_:[] for _ in clusters}
    for idx, lb in enumerate(labels_):
        cluster_items[lb].append(idx)

    # judge label type of the cluster
    print(len(clusters))
    cluster_true_label_map = {}
    for cluster, idxs in cluster_items.items():
        label_cnt = {_:0 for _ in true_types}
        for idx in idxs:
            label_cnt[true_labels[idx]] += 1
        label_sort = sorted(label_cnt.items(), key=lambda d: d[1], reverse=True)
        print(label_sort)
        cluster_true_label_map[cluster] = label_sort[0][0]
    
    return clusters, true_types, cluster_true_label_map


def calc_precision_and_confusion(true_labels, labels_, true_types, clusters, cluster_true_label_map):
    cluster_idx = {_:idx  for idx, _ in enumerate(clusters)}
    type_idx = {_:idx  for idx, _ in enumerate(true_types)}    
    confu_mat = np.zeros((len(clusters), len(true_types)), dtype=np.int32)
    for idx, cluster in enumerate(labels_):
        confu_mat[cluster_idx[cluster]][type_idx[true_labels[idx]]] += 1
    print(confu_mat)

    confu_mat = np.zeros((len(true_types), len(true_types)), dtype=np.int32)
    ypred = []
    for idx, cluster in enumerate(labels_):
        confu_mat[ type_idx[cluster_true_label_map[cluster]]][type_idx[true_labels[idx]]] += 1
        ypred.append(cluster_true_label_map[cluster])
    print(confu_mat)

    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, ypred, labels=true_types)
    # Only use the labels that appear in the data
    uniq = unique_labels(true_labels, ypred)
    
    normalize = True
    cmap = plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    classes = true_types
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # Show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # And label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    # fig.savefig("main_confusion.pdf", bbox_inches='tight')
    
    if len(clusters) == 2 and len(true_types) == 2:
        average = 'binary'
    else:
        average = 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support([type_idx[_] for _ in true_labels], [type_idx[_] for _ in ypred], average=average)
    acc = accuracy_score(true_labels, ypred)
    res = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    return res


sciq_ds = [' '.join(json.loads(_)['text']).lower() for _ in open(file_paths_new['SciQ'])]
race_ds = [' '.join(json.loads(_)['text']).lower() for _ in open(file_paths_new['RACE'])]
nq_ds = [' '.join(json.loads(_)['text']).lower() for _ in open(file_paths_new['NQ'])]


pca_dims = 100
corpus = {
    'NQ': nq_ds,
    'RACE': race_ds,
    'SciQ': sciq_ds,
}


if os.path.exists(f'features-{models_to_use[0]}.race-sicq-targetset.pt'):
    features = torch.load(f'features-{models_to_use[0]}.race-sicq-targetset.pt')
else:
    features = {_:[] for _ in corpus.keys()}
    model = AutoModel.from_pretrained(models_to_use[0]).eval().to(1)
    tokenizer = AutoTokenizer.from_pretrained(models_to_use[0])
    with torch.no_grad():
        for domain, ds in corpus.items():
            # items = list(set(ds))
            items = ds
            avg_pool = []
            cls_features = []
            avg_all_pool = []
            for d in tqdm(items):
                out = model(**tokenizer(d, max_length=512, truncation=True, return_tensors='pt').to(1), output_hidden_states=True)
                average_pool = torch.mean(out[0], dim=1).detach().cpu()
                cls_pooler = out[1].detach().cpu()
                average_pool_all_layers = torch.mean( torch.cat([torch.mean(_, dim=1).detach().cpu() for _ in out[2]]), dim=0).unsqueeze(dim=0)
            
                avg_pool.append(average_pool)
                cls_features.append(cls_pooler)
                avg_all_pool.append(average_pool_all_layers)
            features[domain].append( torch.cat(avg_pool, dim=0))
            features[domain].append( torch.cat(cls_features, dim=0))
            features[domain].append( torch.cat(avg_all_pool, dim=0))
    torch.save(features, f'features-{models_to_use[0]}.race-sicq-targetset.pt')


def make_clustering( corpus_used=['NQ', 'RACE', 'SciQ'], num_samples_for_clustering = 3000):
    trial_name = '-'.join(corpus_used)

    states_used = {
        'avg': 0,
        'cls': 1,
        'all': 2
    }

    
    start_idx = 0
    corpus_idx = dict()
    true_labels = []
    idx_for_clustering = {}    
    for _ in corpus_used:
        corpus_idx[_] = start_idx
        start_idx = start_idx + features[_][0].shape[0]
        true_labels.extend( [_]*features[_][0].shape[0] )
    
        items = corpus[_]
        selected_items = set()
        selected_idx = []
        shuffled_idx = np.arange(len(items))
        np.random.shuffle(shuffled_idx)
        for idx in shuffled_idx:
            if items[idx] not in selected_items:
                selected_items.add(items[idx])
                selected_idx.append(idx)
        idx_for_clustering[_] = np.array( selected_idx )

    num_items_used = { _:min(num_samples_for_clustering, len(idx_for_clustering[_])) for _ in corpus_used }

    final_results = {}
    for pool_type, pool_states_id in states_used.items():
        final_results[pool_type] = {}
        pool_states = torch.cat([features[_][pool_states_id] for _ in corpus_used ], dim=0)
        pca_features = PCA(n_components=pca_dims, random_state=0).fit_transform( pool_states )

        print(f"states shape: {pca_features.shape}")

        states_for_clustering = []
        
        for _ in corpus_used:
            print(f"corpus range: {corpus_idx[_]}, : {features[_][0].shape[0]}")
            _ftr = pca_features[corpus_idx[_] : corpus_idx[_] + features[_][0].shape[0], :]
            print(f"corpus {_} feature shape: {_ftr.shape}, idx_for_clustering shape: {idx_for_clustering[_].shape}")
            selected_ftr = _ftr[idx_for_clustering[_][:num_items_used[_]], :]
            print("-----", selected_ftr.shape)
            states_for_clustering.append(selected_ftr)

        states_for_clustering = np.vstack(states_for_clustering)
        len_ = np.sqrt((states_for_clustering**2).sum(axis=1))[:, None]
        print("************========***", len_.shape)
        print("************========***", states_for_clustering.shape)
        cosine_normalized_states = states_for_clustering / len_
        true_labels_for_clustering = []
        for _ in corpus_used:
            true_labels_for_clustering.extend([_]*num_items_used[_])
        
        print(f"states for clustering shape: {states_for_clustering.shape}")

        n_components = len(corpus_used)
        # GMM
        gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', max_iter=150, random_state=0).fit(states_for_clustering)
        c_pred = gmm.predict(states_for_clustering)

        # GMM with Cosine
        gmm_cosine = GaussianMixture(n_components=n_components, covariance_type='spherical', max_iter=150, random_state=0).fit(cosine_normalized_states)
        len_ = np.sqrt(np.square(gmm_cosine.means_**2).sum(axis=1)[:,None])
        print("========***============", len_.shape)
        print("************========***", gmm_cosine.means_.shape)
        gmm_cosine_centers = gmm_cosine.means_ / len_
        g_pred = gmm_cosine.predict(cosine_normalized_states)
        len_ = np.sqrt(np.square(pca_features**2).sum(axis=1)[:,None])
        cosine_pca_features = pca_features / len_
        # assign types for clusters first.        
        clusters, true_types, cluster_true_label_map = assign_cluster_label_with_type(true_labels_for_clustering, c_pred)
        cluser_metrics = calc_precision_and_confusion(true_labels_for_clustering, c_pred, true_types, clusters, cluster_true_label_map)
        print('gmm:', pool_type, trial_name, cluser_metrics)
        plt.savefig(f"clustering_{pool_type}_confusion_gmm_{trial_name}.target.pdf", dpi=600, bbox_inches='tight')

        cosine_clusters, cosine_true_types, cosine_cluster_true_label_map = assign_cluster_label_with_type(true_labels_for_clustering, g_pred)
        cosine_cluser_metrics = calc_precision_and_confusion(true_labels_for_clustering, g_pred, cosine_true_types, cosine_clusters, cosine_cluster_true_label_map)
        print('cosine gmm:', pool_type, trial_name, cosine_cluser_metrics)
        plt.savefig(f"clustering_{pool_type}_confusion_gmm_cosine_{trial_name}.target.pdf", dpi=600, bbox_inches='tight')

        gmm_pred = gmm.predict(pca_features)
        gmm_proba = gmm.predict_proba(pca_features)
        gmm_scores = gmm.score_samples(pca_features)
        metrics = calc_precision_and_confusion(true_labels, gmm_pred, true_types, clusters, cluster_true_label_map)

        cosine_gmm_pred = gmm_cosine.predict(cosine_pca_features)
        cosine_gmm_proba = gmm_cosine.predict_proba(cosine_pca_features)
        cosine_gmm_scores = gmm_cosine.score_samples(cosine_pca_features)
        cosine_metrics = calc_precision_and_confusion(true_labels, cosine_gmm_pred, cosine_true_types, cosine_clusters, cosine_cluster_true_label_map)

        plt.savefig(f"{pool_type}_confusion_gmm_{trial_name}.target.pdf", dpi=600, bbox_inches='tight')

        nq_center_dis = None
        other_center_dis = None

        nq_center_l2_dis = None
        other_center_l2_dis = None

        if len(corpus_used) == 2:
            other_corpus = None 
            for c in corpus_used:
                if c != 'NQ':
                    other_corpus = c 
            
            label_id_map = { v:k for k, v in cluster_true_label_map.items() }
            if len(label_id_map.keys()) == 1:
                k =  other_corpus if list(label_id_map.keys())[0] == 'NQ' else 'NQ'
                label_id_map[k] =  1 - list(label_id_map.values())[0]
            print('****cluster label map::', corpus_used, cluster_true_label_map, label_id_map)
            race_label_id = label_id_map[other_corpus]
            nq_label_id = label_id_map['NQ']
            print(label_id_map, race_label_id, nq_label_id)        

            gmm_centers = gmm.means_
            nq_gmm_center = gmm_centers[nq_label_id, :]
            race_gmm_center = gmm_centers[race_label_id, :]
            
            nq_gmm_cosine_center = gmm_cosine_centers[nq_label_id, :]
            race_gmm_cosine_center = gmm_cosine_centers[race_label_id, :]
            # cosine_dis
            # nq_center_dis = [distance.cosine(pca_features[_, :], nq_gmm_center) for _ in range(pca_features.shape[0])]
            # other_center_dis = [distance.cosine(pca_features[_, :], race_gmm_center) for _ in range(pca_features.shape[0])] 
            nq_cosin_dist = 1 - np.dot(nq_gmm_cosine_center, cosine_pca_features.T)
            other_cosin_dist = 1 - np.dot(race_gmm_cosine_center, cosine_pca_features.T)
            # euclidean distance
            nq_center_l2_dis = [distance.euclidean(pca_features[_, :], nq_gmm_center) for _ in range(pca_features.shape[0])]
            other_center_l2_dis = [distance.euclidean(pca_features[_, :], race_gmm_center) for _ in range(pca_features.shape[0])] 
            # Jensen-shannon Divergence
            nq_center_jensenshannon_dis = [distance.jensenshannon(pca_features[_, :], nq_gmm_center) for _ in range(pca_features.shape[0])]
            other_center_jensenshannon_dis = [distance.jensenshannon(pca_features[_, :], race_gmm_center) for _ in range(pca_features.shape[0])] 


        final_results[pool_type]['gmm'] = {'means': gmm.means_, 'covar':gmm.covariances_, 'cluster_metrics':cluser_metrics, 'metrics': metrics, 
                    'proba':gmm_proba, 'llscores':gmm_scores, 'corpus_idx':corpus_idx, 'cluster_label_map':cluster_true_label_map, 
                    'cosine_metrics':cosine_metrics, 
                    'nq_center_cosine_distance': nq_cosin_dist, 'other_center_cosine_distance':other_cosin_dist,
                    'nq_center_l2_distance':nq_center_l2_dis , 'other_center_l2_distance':other_center_l2_dis,
                    'nq_center_jensenshannon_distance':nq_center_jensenshannon_dis, 'other_center_jensenshannon_distance':other_center_jensenshannon_dis}   #, f'{pool_type}_gmm_centers_{trial_name}.pt')

        # km = cluster.KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=0).fit(states_for_clustering)
        minikm = cluster.MiniBatchKMeans(n_clusters=n_components, init='k-means++', n_init=1, random_state=0,
                        init_size=1000, batch_size=2000, verbose=0).fit(states_for_clustering)
        
        # assign type for the clusters.
        clusters, true_types, cluster_true_label_map = assign_cluster_label_with_type(true_labels_for_clustering, minikm.labels_)
        cluster_metrics = calc_precision_and_confusion(true_labels_for_clustering, minikm.labels_, true_types, clusters, cluster_true_label_map)
        print('kmeans cluster metrics:', pool_type, trial_name, cluster_metrics)
        plt.savefig(f"clustering_{pool_type}_confusion_minikm_{trial_name}.target.pdf", dpi=600, bbox_inches='tight')

        minikm_pred = minikm.predict(pca_features)
        distances = minikm.transform(pca_features)
        metrics = calc_precision_and_confusion(true_labels, minikm_pred, true_types, clusters, cluster_true_label_map)
        print('kmeans:', pool_type, trial_name, metrics)
        plt.savefig(f"{pool_type}_confusion_minikm_{trial_name}.target.pdf", dpi=600, bbox_inches='tight')
        final_results[pool_type]['kmeans'] = {'centers': minikm.cluster_centers_, 'cluster_metrics':cluster_metrics, 'metrics': metrics, 'distances':distances, 'corpus_idx':corpus_idx, 'cluster_label_map':cluster_true_label_map}
    final_results['idx_for_clustering'] = idx_for_clustering
    final_results['num_samples_for_clustering'] = num_items_used
    torch.save(final_results, f'results_{trial_name}.target.pt')


make_clustering(corpus_used=['NQ', 'RACE'], num_samples_for_clustering=3000)

# make_clustering(corpus_used=['NQ', 'SciQ'], num_samples_for_clustering=5000)

# make_clustering(num_samples_for_clustering=2000)