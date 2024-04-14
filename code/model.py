import time
import world
import torch

import dataloader
from dataloader import BasicDataset
from torch import nn, multiprocessing
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)


class CIPS(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(CIPS, self).__init__()
        self.Po_items_mean = None
        self.Po_users_mean = None
        self.In_items_mean = None
        self.In_users_mean = None
        self.Po_items = None
        self.Po_users = None
        self.In_items = None
        self.In_users = None
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.activation = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):

        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items

        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_user1 = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item1 = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc1 = nn.Linear(self.latent_dim, 1)
        self.fc2 = nn.Linear(self.latent_dim, 1)
        self.fc3 = nn.Linear(self.latent_dim, 1)
        self.fc4 = nn.Linear(self.latent_dim, 1)

        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            nn.init.normal_(self.embedding_user1.weight, std=0.1)
            nn.init.normal_(self.embedding_item1.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            self.embedding_user1.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item1.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.Graph2 = self.dataset.getSparseGraph2()
        print(f"cips is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout_x_2(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def __dropout2(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph2:
                graph.append(self.__dropout_x_2(g, keep_prob))
        else:
            graph = self.__dropout_x_2(self.Graph2, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for CIPS
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                #                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            #        for layer in range(0):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        #        print(embs.shape)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def computer2(self):
        """
        propagate methods for CIPS
        """
        users_emb = self.embedding_user1.weight
        items_emb = self.embedding_item1.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                #                print("droping")
                g_droped = self.__dropout2(self.keep_prob)
            else:
                g_droped = self.Graph2
        else:
            g_droped = self.Graph2

        for layer in range(self.n_layers):
            #        for layer in range(0):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):

        # time0 = time.perf_counter()
        all_users, all_items = self.computer2()
        all_users2, all_items2 = self.computer()
        # time1 = time.perf_counter()
        # print("computer time:", time1 - time0)
        #

        ##CIPS

        w1 = torch.sigmoid(self.fc1(all_users) + self.fc2(all_users2))
        w2 = torch.sigmoid(self.fc3(all_items) + self.fc4(all_items2))

        w1 = self.dataset.users_cnt * (1 - self.config["lam1"]) + w1 * self.config["lam1"]
        w2 = self.dataset.items_cnt * (1 - self.config["lam2"]) + w2 * self.config["lam2"]

        all_users = torch.stack((all_users * w1, all_users2 * (1 - w1)), -1)
        all_items = torch.stack((all_items * w2, all_items2 * (1 - w2)), -1)
        all_users = torch.sum(all_users, -1)
        all_items = torch.sum(all_items, -1)

        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        # compute embedding
        # time0 = time.perf_counter()
        all_users, all_items = self.computer2()
        all_users2, all_items2 = self.computer()
        # time1 = time.perf_counter()
        # print("computer time:", time1 - time0)


        w1 = torch.sigmoid(self.fc1(all_users) + self.fc2(all_users2))
        w2 = torch.sigmoid(self.fc3(all_items) + self.fc4(all_items2))

        w1 = self.dataset.users_cnt * (1 - self.config["lam1"]) + w1 * self.config["lam1"]
        w2 = self.dataset.items_cnt * (1 - self.config["lam2"]) + w2 * self.config["lam2"]

        all_users = torch.stack((all_users * w1, all_users2 * (1 - w1)), -1)
        all_items = torch.stack((all_items * w2, all_items2 * (1 - w2)), -1)
        all_users = torch.sum(all_users, -1)
        all_items = torch.sum(all_items, -1)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        users_emb_ego1 = self.embedding_user1(users)
        pos_emb_ego1 = self.embedding_item1(pos_items)
        neg_emb_ego1 = self.embedding_item1(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, users_emb_ego1, pos_emb_ego1, neg_emb_ego1

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0,
         userEmb1, posEmb1, negEmb1) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        reg_loss += (1 / 2) * (userEmb1.norm(2).pow(2) +
                               posEmb1.norm(2).pow(2) +
                               negEmb1.norm(2).pow(2)) / float(len(users))
        reg_loss += 5 * ((userEmb1 - userEmb0).norm(2).pow(2) +
                         (posEmb1 - posEmb0).norm(2).pow(2) +
                         (negEmb1 - negEmb0).norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        # time0 = time.perf_counter()
        all_users, all_items = self.computer2()
        all_users2, all_items2 = self.computer()
        # time1 = time.perf_counter()
        # print("computer time:", time1 - time0)

        w1 = torch.sigmoid(self.fc1(all_users) + self.fc2(all_users2))
        w2 = torch.sigmoid(self.fc3(all_items) + self.fc4(all_items2))

        w1 = self.dataset.users_cnt * (1 - self.config["lam1"]) + w1 * self.config["lam1"]
        w2 = self.dataset.items_cnt * (1 - self.config["lam2"]) + w2 * self.config["lam2"]

        all_users = torch.stack((all_users * w1, all_users2 * (1 - w1)), -1)
        all_items = torch.stack((all_items * w2, all_items2 * (1 - w2)), -1)
        all_users = torch.sum(all_users, -1)
        all_items = torch.sum(all_items, -1)

        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

