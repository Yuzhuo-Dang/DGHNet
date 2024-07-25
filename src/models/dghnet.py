import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence
from torch.nn.functional import softplus
from torch.autograd import Variable

from src.common.abstract_recommender import GeneralRecommender

class DGHNet(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DGHNet, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim'] #模态特征维度64维
        self.knn_k = config['knn_k'] #10
        self.mm_bpr_loss_weight = config['mm_bpr_loss_weight']
        self.vae_loss_weight = config['vae_loss_weight']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers'] #卷积层数1
        self.n_ui_layers = config['n_ui_layers'] #卷积层数2
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight'] #图特征融合时图像模态的权重
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']

        self.n_nodes = self.n_users + self.n_items
        self.random = config['seed']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.fusion_layer = nn.Linear(64 * 2, 64).to(self.device)
        #self.cat_layer = nn.Linear(64 * 4, 64).to(self.device)

        self.att_fc1 = nn.Linear(64, 16)
        self.att_fc2 = nn.Linear(16, 2)

        #计算D
        self.vae_net_image = nn.Sequential(
            nn.Linear(self.v_feat.shape[1], 64),
            nn.ReLU(True),
            nn.Linear(64, 4),
        )
        self.vae_net_text = nn.Sequential(
            nn.Linear(self.t_feat.shape[1], 64),
            nn.ReLU(True),
            nn.Linear(64, 4),
        )
        self.vae_net_decoder_image = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, self.v_feat.shape[1]),
        )
        self.vae_net_decoder_text = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, self.t_feat.shape[1]),
        )
        self.sigmoid = nn.Sigmoid()

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_mmdsp_{}.pt'.format(self.knn_k))
        text_adj_file = os.path.join(dataset_path, 'text_adj_mmdsp_{}.pt'.format(self.knn_k))

        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim).to(self.device)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim).to(self.device)

        if os.path.exists(image_adj_file) and os.path.exists(text_adj_file):
            self.image_adj = torch.load(image_adj_file)
            self.text_adj = torch.load(text_adj_file)
        else:
            indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            self.image_adj = image_adj
            torch.save(self.image_adj, image_adj_file)
            indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            self.text_adj = text_adj
            torch.save(self.text_adj, text_adj_file)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def fusion_mm(self):
        image_feats = F.leaky_relu(self.image_trs(self.image_embedding.weight))
        text_feats = F.leaky_relu(self.text_trs(self.text_embedding.weight))
        combined_features = torch.cat((image_feats, text_feats), dim=1).to(self.device)
        fusion_output = F.tanh(self.fusion_layer(combined_features)).to(self.device)

        _, mm_adj = self.get_knn_adj_mat(fusion_output)
        return fusion_output, mm_adj

    #计算模糊度
    def econder(self, x):
        if x.shape[1] == self.v_feat.shape[1]:
            params = self.vae_net_image(x)
            mu, sigma = params[:, :2], params[:, 2:]
            self.mu_image = mu
            self.sigma_image = sigma
        else:
            params = self.vae_net_text(x)
            mu, sigma = params[:, :2], params[:, 2:]
            self.mu_text = mu
            self.sigma_text = sigma
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1) #变分自编码器
    def d(self):
        p_z1_given_image= self.econder(self.image_embedding.weight)
        p_z2_given_text = self.econder(self.text_embedding.weight)
        kl12 = kl_divergence(p_z1_given_image, p_z2_given_text)
        kl21 = kl_divergence(p_z2_given_text, p_z1_given_image)
        d = (kl12 + kl21)/ 2.
        d = nn.functional.tanh(d)
        return d

    def forward(self, adj):
        fusion_out, mm_adj = self.fusion_mm()
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.sparse.mm(mm_adj, h)

        h_v = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h_v = torch.sparse.mm(self.image_adj, h_v)

        h_t = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h_t = torch.sparse.mm(self.text_adj, h_t)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        d = self.d()# torch.mean(self.d())
        d = d.unsqueeze(1).expand(-1, h.size(1))

        #d1 = torch.full((self.n_items,64), 0.5).to(self.device)
        h1 = torch.mul(self.mm_image_weight * h_v + (1-self.mm_image_weight) * h_t, 1-d) #文本和图像先融合
        h2 = torch.mul(h, d) #融合的
        h3 = i_g_embeddings

        return u_g_embeddings, h1 * 2 + h2 + h3 #item_embeddings 超过基线模型了i_g_embeddings + 0.5 * h + 0.2 * h_v + 0.8 * h_t
        # h1 * 2 + h2 + h3

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def vae_loss(self):
        std1 = self.sigma_image.mul(0.5).exp_()
        eps1 = Variable(std1.data.new(std1.size()).normal_())
        z1 = eps1.mul(std1).add_(self.mu_image)
        std2 = self.sigma_text.mul(0.5).exp_()
        eps = Variable(std2.data.new(std2.size()).normal_())
        z2 = eps.mul(std2).add_(self.mu_text)

        h1 = self.sigmoid(self.vae_net_decoder_image(z1))
        h2 = self.sigmoid(self.vae_net_decoder_text(z2))

        BCE_image = F.binary_cross_entropy(h1, self.sigmoid(self.image_embedding.weight), reduction='mean')
        BCE_text = F.binary_cross_entropy(h2, self.sigmoid(self.text_embedding.weight), reduction='mean')

        KLD_image = -0.5 * torch.sum(1 + self.sigma_image - self.mu_image.pow(2) - self.sigma_image.exp()) / self.image_embedding.weight.size()[0]
        KLD_text = -0.5 * torch.sum(1 + self.sigma_text - self.mu_text.pow(2) - self.sigma_text.exp()) / self.text_embedding.weight.size()[0]

        return BCE_image + BCE_text + KLD_image + KLD_text

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, item_embedding = self.forward(self.masked_adj) #self.masked_adj
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = item_embedding[pos_items]
        neg_i_g_embeddings = item_embedding[neg_items]
        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        fusion_output, _, = self.fusion_mm()
        pos_i_g_embeddings1 = fusion_output[pos_items]
        neg_i_g_embeddings1 = fusion_output[neg_items]
        batch_mf_loss_mm = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings1, neg_i_g_embeddings1)

        vae_loss = self.vae_loss()
        return batch_mf_loss + self.mm_bpr_loss_weight*batch_mf_loss_mm + self.vae_loss_weight*vae_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def item_embedding(self):
        _, item = self.forward(self.masked_adj)
        return item.cpu().detach().numpy()
