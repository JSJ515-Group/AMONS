import torch
import torch.nn as nn
import os
import numpy as np
import time
import scipy.sparse as sp
from scipy import sparse
from losses import compute_info_bpr_loss, compute_l2_loss
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.data.sampler import BatchSampler

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

dataset_base_path = '../data/gowalla'
user_num = 29858
item_num = 40981
factor_num = 64
batch_size = 8000
top_k = 20
num_negative_test_val = -1
n_fold = 20
num_negs=300

run_id = "g"
print(run_id)
dataset = 'gowalla'
path_save_base = './log/' + dataset + '/newloss' + run_id
result_file = open(path_save_base + '/results.txt', 'a')
path_save_model_base = '../newlossModel/' + dataset + '/s' + run_id

training_user_set, training_item_set, training_set_count = np.load(dataset_base_path + '/datanpy/training_set.npy', allow_pickle=True)
testing_user_set, testing_item_set, testing_set_count = np.load(dataset_base_path + '/datanpy/testing_set.npy', allow_pickle=True)
user_rating_set_all = np.load(dataset_base_path + '/datanpy/user_rating_set_all.npy', allow_pickle=True).item()
sparse_user_item_adj = sparse.load_npz(dataset_base_path + '/datanpy/s_norm_adj_mat.npz')
combined_user_item_dict = training_user_set
def readD(set_matrix, num_):
    user_d = []
    for i in range(num_):
        len_set = 1.0 / (len(set_matrix[i]) + 1)
        user_d.append(len_set)
    return user_d

u_d = readD(training_user_set, user_num)
i_d = readD(training_item_set, item_num)
d_i_train = u_d
d_j_train = i_d

def readTrainSparseMatrix(set_matrix, is_user):
    user_items_matrix_i = []
    user_items_matrix_v = []
    if is_user:
        d_i = u_d
        d_j = i_d
    else:
        d_i = i_d
        d_j = u_d
    for i in set_matrix:
        len_set = len(set_matrix[i])
        for j in set_matrix[i]:
            user_items_matrix_i.append([i, j])
            d_i_j = np.sqrt(d_i[i] * d_j[j])
            user_items_matrix_v.append(d_i_j)
    user_items_matrix_i = torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v = torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

sparse_u_i = readTrainSparseMatrix(training_user_set, True)
sparse_i_u = readTrainSparseMatrix(training_item_set, False)


class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num, user_item_matrix, item_user_matrix, d_i_train, d_j_train, norm_adj):
        super(BPR, self).__init__()
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.norm_adj = norm_adj
        self.embed_user = nn.Embedding(user_num, factor_num).cuda()
        self.embed_item = nn.Embedding(item_num, factor_num).cuda()

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

    def _split_A_hat(self, X):
        n_fold = 20
        user_num = 29858
        item_num = 40981
        A_fold_hat = []
        fold_len = (user_num + item_num) // n_fold
        for i_fold in range(n_fold):
            start = i_fold * fold_len
            if i_fold == n_fold - 1:
                end = user_num + item_num
            else:
                end = (i_fold + 1) * fold_len
            coo = sp.coo_matrix(X[start:end]).astype(np.float32)
            merged_array = np.stack((coo.row, coo.col, coo.data), axis=1)
            indices = torch.tensor(merged_array[:, :2], dtype=torch.long).to('cuda')
            values = torch.tensor(merged_array[:, 2], dtype=torch.float32).to('cuda')
            shape = torch.Size(coo.shape)
            A_fold_hat.append(torch.sparse.FloatTensor(indices.t(), values, shape).to('cuda'))
        return A_fold_hat

    def create_tensor_dataloader(self, tensor, batch_size):
        dataset = TensorDataset(tensor)
        sampler = RandomSampler(dataset)
        return DataLoader(dataset,sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False),
                          collate_fn=lambda batchs: batchs[0][0])

    def forward(self, n_fold, num_negs):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight
        ego_embeddings = torch.cat([users_embedding, items_embedding], axis=0).cuda()
        temp_embed = []
        for f in range(n_fold):
            temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings))
        users_items_embedding = torch.cat(temp_embed, 0) + ego_embeddings
        users_embedding, items_embedding = torch.split(users_items_embedding, [user_num, item_num], dim=0)

        gcn1_users_embedding = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train)).cuda()
        gcn1_items_embedding = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train)).cuda()

        gcn2_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train)).cuda()
        gcn2_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train)).cuda()

        gcn3_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) + gcn2_users_embedding.mul(
            self.d_i_train)).cuda()
        gcn3_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) + gcn2_items_embedding.mul(
            self.d_j_train)).cuda()

        gcn4_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn3_items_embedding) + gcn3_users_embedding.mul(
            self.d_i_train)).cuda()
        gcn4_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn3_users_embedding) + gcn3_items_embedding.mul(
            self.d_j_train)).cuda()

        gcn_users_embedding = users_embedding + gcn1_users_embedding * (1 / 2) + gcn2_users_embedding * (
                    1 / 3) + gcn3_users_embedding * (1 / 4) + gcn4_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding * (1 / 2) + gcn2_items_embedding * (
                    1 / 3) + gcn3_items_embedding * (1 / 4) + gcn4_items_embedding

        edges = []
        with open(dataset_base_path + '/train.txt', "r", encoding="utf-8") as f:
            for l in f.readlines():
                if len(l) > 0:
                    try:
                        l = l.strip('\n').split(' ')
                        items = []
                        uid = int(l[0])
                        for i in l[1:]:
                            i = int(i)
                            items.append(i)
                            edges.append([uid, i])
                    except Exception:
                        continue
        edges = np.array(edges)
        train_edges_data_loader = self.create_tensor_dataloader(torch.tensor(edges), batch_size)
        for step, batch_edges in enumerate(train_edges_data_loader):
            mf_losses = compute_info_bpr_loss(gcn_users_embedding, gcn_items_embedding, batch_edges, combined_user_item_dict, num_negs, reduction="none")
            l2_loss = compute_l2_loss([users_embedding, items_embedding])
            loss = mf_losses.sum() + l2_loss * 1e-4
        return loss

model = BPR(user_num, item_num, factor_num, sparse_u_i, sparse_i_u, d_i_train, d_j_train, sparse_user_item_adj)
model = model.to('cuda')
optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.001)

print('--------training processing-------')
for epoch in range(2000):
    model.train()
    start_time = time.time()
    model.zero_grad()
    loss= model(n_fold, num_negs)
    loss.backward()
    optimizer_bpr.step()
    elapsed_time = time.time() - start_time
    str_print_train = "epoch:" + str(epoch) + '\t time:' + str(round(elapsed_time, 1)) + '\t train loss:' + str(loss)
    PATH_model = path_save_model_base + '/epoch' + str(epoch) + '.pt'
    torch.save(model.state_dict(), PATH_model)
    print(str_print_train)
    result_file.write("epoch:" + str(epoch) + '\t time:' + str(round(elapsed_time, 1)) + '\t train loss:' + str(loss))
    result_file.write('\n')
    result_file.flush()