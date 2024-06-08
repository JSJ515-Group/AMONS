import scipy.sparse as sp
from scipy import sparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import evaluate
import data_utils
from torch.utils.data import DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

dataset_base_path = '../data/amazon-book'

user_num = 52643
item_num = 91599
factor_num = 64
batch_size = 8000
top_k = 20
num_negative_test_val = -1
start_i_test = 1500
end_i_test = 2000
setp = 100

run_id = "a"
dataset = 'amazon-book'
path_save_base = './log/' + dataset + '/newloss' + run_id
result_file = open(path_save_base + '/results_hdcg_hr.txt', 'a')
path_save_model_base = '../newlossModel/' + dataset + '/s' + run_id

training_user_set, training_item_set, training_set_count = np.load(dataset_base_path + '/datanpy/training_set.npy',
                                                                   allow_pickle=True)
testing_user_set, testing_item_set, testing_set_count = np.load(dataset_base_path + '/datanpy/testing_set.npy',
                                                                allow_pickle=True)
user_rating_set_all = np.load(dataset_base_path + '/datanpy/user_rating_set_all.npy', allow_pickle=True).item()
sparse_user_item_adj = sparse.load_npz(dataset_base_path + '/datanpy/s_norm_adj_mat.npz')

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
            user_items_matrix_v.append(d_i_j)  # (1./len_set)
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
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        self.norm_adj = norm_adj

        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def _split_A_hat(self, X):
        n_fold = 40
        user_num = 52643
        item_num = 91599
        A_fold_hat = []
        fold_len = (user_num + item_num) // n_fold
        for i_fold in range(n_fold):
            start = i_fold * fold_len
            if i_fold == n_fold - 1:
                end = user_num + item_num
            else:
                end = (i_fold + 1) * fold_len
            coo = sp.coo_matrix(X.tocsr()[start:end]).astype(np.float32)
            merged_array = np.stack((coo.row, coo.col, coo.data), axis=1)
            indices = torch.tensor(merged_array[:, :2], dtype=torch.long).to('cuda')
            values = torch.tensor(merged_array[:, 2], dtype=torch.float32).to('cuda')
            shape = torch.Size(coo.shape)
            A_fold_hat.append(torch.sparse.FloatTensor(indices.t(), values, shape).to('cuda'))
        return A_fold_hat

    def forward(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight
        ego_embeddings = torch.cat([users_embedding, items_embedding], axis=0)

        temp_embed = []
        for f in range(40):
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

        gcn_users_embedding = users_embedding + gcn1_users_embedding * (1 / 2) + gcn2_users_embedding * (1 / 3) + gcn3_users_embedding * (1 / 4) + gcn4_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding * (1 / 2) + gcn2_items_embedding * (1 / 3) + gcn3_items_embedding * (1 / 4) + gcn4_items_embedding
        return gcn_users_embedding, gcn_items_embedding

test_batch = 52
testing_dataset = data_utils.resData(train_dict=testing_user_set, batch_size=test_batch, num_item=item_num,
                                     all_pos=training_user_set)
testing_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=0)


model = BPR(user_num, item_num, factor_num, sparse_u_i, sparse_i_u, d_i_train, d_j_train, sparse_user_item_adj)
model = model.to('cuda')
optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.001)


def largest_indices(ary, n):
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


print('--------test processing-------')
for epoch in range(start_i_test, end_i_test, setp):
    model.train()

    PATH_model = path_save_model_base + '/epoch' + str(epoch) + '.pt'
    print(PATH_model)
    model.load_state_dict(torch.load(PATH_model))
    model.eval()
    gcn_users_embedding, gcn_items_embedding = model()
    user_e = gcn_users_embedding.detach().cpu().numpy()
    item_e = gcn_items_embedding.detach().cpu().numpy()

    all_pre = np.matmul(user_e, item_e.T)
    HR, NDCG = [], []
    set_all = set(range(item_num))
    test_start_time = time.time()
    for u_i in testing_user_set:
        item_i_list = list(testing_user_set[u_i])
        index_end_i = len(item_i_list)
        item_j_list = list(set_all - training_user_set[u_i] - testing_user_set[u_i])
        item_i_list.extend(item_j_list)

        pre_one = all_pre[u_i][item_i_list]
        indices = largest_indices(pre_one, top_k)
        indices = list(indices[0])

        hr_t, ndcg_t = evaluate.hr_ndcg(indices, index_end_i, top_k)
        elapsed_time = time.time() - test_start_time
        HR.append(hr_t)
        NDCG.append(ndcg_t)
    hr_test = round(np.mean(HR), 4)
    ndcg_test = round(np.mean(NDCG), 4)

    str_print_evl = "epoch:" + str(epoch) + '\t time:' + str(
        round(elapsed_time, 2)) + "\t ---test---" + "\t hit:" + str(
        hr_test) + '\t ndcg:' + str(ndcg_test)
    print(str_print_evl)

    result_file.write(str_print_evl)
    result_file.write('\n')
    result_file.flush()