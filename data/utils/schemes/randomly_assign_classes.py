import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def randomly_assign_classes(
    dataset: Dataset, client_num: int, class_num: int
) -> Tuple[List[List[int]], Dict[str, Dict[str, int]]]:
    partition = {"separation": None, "data_indices": None}
    data_indices = [[] for _ in range(client_num)]
    # targets_numpy: np.array of labels
    targets_numpy = np.array(dataset.targets, dtype=np.int32)
    # label_list: a list from 0 to 9 for example
    label_list = list(range(len(dataset.classes)))
    # data_idx_for_each_label: a list where elements are the indices of
    # pictures from class i
    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0].tolist() for i in label_list
    ]
    assigned_labels = []
    # selected_times: counting list of occurance 
    selected_times = [0 for _ in label_list]
    for i in range(client_num):
        # sampled_labels: randomly select ''class_num'' elements from label_list
        sampled_labels = random.sample(label_list, class_num)
        # assigned_labels: a list of the class distribution at a client
        assigned_labels.append(sampled_labels)
        for j in sampled_labels:
            # class selected counting time +1
            selected_times[j] += 1
    
    # labels_count: a dictionary where keys are labels, 
    #                                  values are occurance times.
    labels_count = Counter(targets_numpy)
    # batch_size: np.array() of size label numbers
    batch_sizes = np.zeros_like(label_list)
    for i in label_list:
        # batch_sizes[i] is the slice size of one selection, 
        #                              which might be uneven.
        # note that batch size is rounded.
        batch_sizes[i] = int(labels_count[i] / selected_times[i])
    
    # need a step to uniform clients' volume
    client_volume = []
    client_volume_list = []
    for i in range(client_num):
        c_volume = 0
        c_volume_list = []
        # It is possible that there are more than 2 labels.
        for j in assigned_labels[i]:
            c_volume += batch_size[j]
            c_volume_list.append(batch_size[j])

        client_volume.append(c_volume)
        client_volume_list.append(np.array(c_volume_list))
    
    client_min_volume = np.array(client_volume).min()

    client_volume_list_normalized = []

    for i in range(client_num):
        if client_volume[i] > client_min_volume:
            # raw scaled volumes
            c_volume_list_normalized = \
                client_min_volume / client_volume[i] * client_volume_list[i]
            # rounded scaled volumes
            c_volume_list_normalized = np.ceil(c_volume_list_normalized)
            # but it is possible they do not add up to min_volume
            # however, the gap cannot exceed class_num by simple math
            gap = client_min_volume - c_volume_list_normalized.sum()
            ref_ind = random.sample(list(range(class_num)), gap)                
            c_volume_list_normalized[ref_ind] += 1
            client_volume_list_normalized.append(c_volume_list_normalized)
        else:
            # unnormalized client size
            client_volume_list_normalized.append(client_volume_list[i])

        for cls_ind, cls in enumerate(assigned_labels[i]):
            # if overall the volume of pictures from one class <
            # 2 * one slice size.
            # because of the previous step normalization,
            # the "if" clause means that it is the last batch.
            if len(data_idx_for_each_label[cls]) < 2 * batch_sizes[cls]:
                # why not batch_size?
                # this is because batch size has been rounded
                # and might not match the remainig dataset size at last step.
                batch_size = len(data_idx_for_each_label[cls])
            else:
                batch_size = batch_sizes[cls]
            selected_idx = random.sample(data_idx_for_each_label[cls], batch_size)
            # to account for the minimum client dataset volume
            # selected_idx need to shrink in size.
            shrink_size = client_volume_list_normalized[i][cls_ind]
            select_idx = selected_idx[:shrink_size]

            data_indices[i] = np.concatenate(
                [data_indices[i], select_idx], axis=0
            ).astype(np.int64)
            data_idx_for_each_label[cls] = list(
                set(data_idx_for_each_label[cls]) - set(selected_idx)
            )
            # Still, it remains to be unresolved that
            # the local datasets might be unbalanced in volume.
            # Based on the current implementation,
            # a client's local data volume is uncontrollable.

        data_indices[i] = data_indices[i].tolist()

    stats = {}
    for i, idx in enumerate(data_indices):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(idx)
        stats[i]["y"] = Counter(targets_numpy[idx].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices

    return partition, stats
