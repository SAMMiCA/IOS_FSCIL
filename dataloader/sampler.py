import torch, numpy as np, random, copy
from collections import defaultdict


class PairBatchSampler():
    def __init__(self, label, batch_size):
        self.label = np.array(label)  # all data label
        self.batch_size = batch_size
        self.datalen = len(label)
        self.n_batch = (self.datalen+self.batch_size-1) // self.batch_size

        self.m_ind = defaultdict(list)
        for i in range(len(label)):
            y = label[i]
            self.m_ind[y].append(i)

    def __iter__(self):
        indices = list(range(self.datalen))
        random.shuffle(indices)
        for k in range(self.n_batch):
            offset = k*self.batch_size
            batch_indices = indices[offset:offset+self.batch_size]

            pair_indices = []
            for idx in batch_indices:
                y = self.label[idx]
                pair_indices.append(random.choice(self.m_ind[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        return self.n_batch

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per,):
        self.n_batch = n_batch  # the number of iterations in the dataloader. == #episode
        self.n_cls = n_cls
        self.n_per = n_per


        label = np.array(label)  # all data label
        label_list = np.unique(label)

        self.m_ind = []  # the data index of each class
        for i in label_list:
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        """
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        """

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
