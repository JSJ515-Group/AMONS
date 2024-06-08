import torch
import numpy as np
import torch.nn.functional as F

def compute_info_bpr_loss(a_embeddings, b_embeddings, pos_edges, combined_user_item_dict, num_negs, reduction='mean'):
    if isinstance(pos_edges, list):
        pos_edges = np.array(pos_edges)
    device = a_embeddings.device
    a_indices = pos_edges[:, 0]
    b_indices = pos_edges[:, 1]
    num_pos_edges = len(pos_edges)
    num_b = b_embeddings.size(0)
    neg_b_indices = generate_negative_samples(num_pos_edges, num_negs, combined_user_item_dict, num_b)
    a_indices = a_indices.long().to(device)
    b_indices = b_indices.long().to(device)
    embedded_a = a_embeddings[a_indices]
    embedded_b = b_embeddings[b_indices]
    embedded_neg_b = b_embeddings[neg_b_indices]
    embedded_combined_b = torch.cat([embedded_b.unsqueeze(1), embedded_neg_b], 1)
    logits = (embedded_combined_b @ embedded_a.unsqueeze(-1)).squeeze(-1)
    info_bpr_loss = F.cross_entropy(logits, torch.zeros([num_pos_edges], dtype=torch.int64).to(device),
                                    reduction=reduction)
    return info_bpr_loss

def compute_l2_loss(params):
    l2_loss = 0.0
    for param in params:
        l2_loss += param.pow(2).sum() * 0.5
    return l2_loss

def generate_negative_samples(num_pos_edges, num_negs, combined_user_item_dict, num_b):
    neg_b_indices = []
    for i in range(num_pos_edges):
        user_id = i
        neg_samples = set()
        while len(neg_samples) < num_negs:
            neg_sample = torch.randint(0, num_b, (num_negs,))
            if not any((user_id, neg) in combined_user_item_dict for neg in neg_sample):
                neg_samples.update(neg_sample)
        neg_b_indices.append(list(neg_samples))
    return torch.tensor(neg_b_indices)