import logging

import numpy as np
import torch
import torch.nn as nn

from .gat import GraphAttentionLayer


class Encoder(torch.nn.Module):
    def __init__(
        self,
        config,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        alpha,
        device,
        use_ptlm=False,
        pretrained_entity_embeddings=None,
        pretrained_relation_embeddings=None,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = 3
        self.rnn_input_size = config.rnn_input_size
        self.num_neighbor = config.num_neighbor
        self.re_input_size = 2 * self.seq_length * self.hidden_size
        self.pre_linear = nn.Linear(config.ptlm_embedding_dim, self.rnn_input_size)
        self.triple_encoder = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.rule_encoder = nn.LSTM(
            self.re_input_size,
            self.re_input_size,
            2,
            batch_first=True,
            bidirectional=False,
        )
        self.device = device
        self.attention = GraphAttentionLayer(
            self.re_input_size,
            self.re_input_size,
            dropout=dropout,
            alpha=alpha,
        )
        if config.use_ptlm == False:
            logging.warning("using random embeddings for entities and relations")
            self.ent_embeddings = nn.Embedding(
                config.total_ent, config.ptlm_embedding_dim
            )
            self.rel_embeddings = nn.Embedding(
                config.total_rel, config.ptlm_embedding_dim
            )
            uniform_range = 6 / np.sqrt(config.ptlm_embedding_dim)
            nn.init.uniform_(self.ent_embeddings.weight, -uniform_range, uniform_range)
            nn.init.uniform_(self.rel_embeddings.weight, -uniform_range, uniform_range)
        else:
            logging.warning("using pretrained embeddings for entities and relations")
            self.ent_embeddings = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_entity_embeddings), freeze=True
            )
            self.rel_embeddings = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_relation_embeddings), freeze=True
            )

    def forward(
        self,
        batch_h,
        batch_r,
        batch_t,
        batch_rule_h,
        batch_rule_r,
        batch_rule_t,
        batch_rule_conf,
    ):
        head = self.ent_embeddings(batch_h)
        relation = self.rel_embeddings(batch_r)
        tail = self.ent_embeddings(batch_t)

        rule_head = self.ent_embeddings(batch_rule_h)
        rule_relation = self.ent_embeddings(batch_rule_r)
        rule_tail = self.ent_embeddings(batch_rule_t)

        head = self.pre_linear(head)
        relation = self.pre_linear(relation)
        tail = self.pre_linear(tail)

        rule_head = self.pre_linear(rule_head)
        rule_relation = self.pre_linear(rule_relation)
        rule_tail = self.pre_linear(rule_tail)

        batch_triples_emb = torch.cat((head, relation), dim=1)
        batch_triples_emb = torch.cat((batch_triples_emb, tail), dim=1)

        # Triple Encoder
        x = batch_triples_emb.view(-1, 3, self.rnn_input_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            self.device
        )
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            self.device
        )
        out, _ = self.triple_encoder(x, (h0, c0))

        # Rule Encoder
        batch_rules_emb = torch.cat([rule_head, rule_relation, rule_tail], dim=1)
        x = batch_rules_emb.view(-1, 3, self.rnn_input_size)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            self.device
        )
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            self.device
        )
        rules, _ = self.triple_encoder(x, (h0, c0))
        rules = rules.reshape(-1, self.re_input_size)
        rules = rules.view(-1, 3, self.re_input_size)
        h0 = torch.zeros(self.num_layers, rules.size(0), self.re_input_size).to(
            self.device
        )
        c0 = torch.zeros(self.num_layers, rules.size(0), self.re_input_size).to(
            self.device
        )
        _, (rules_repr, _) = self.rule_encoder(rules, (h0, c0))

        # GAT
        out = out.reshape(-1, self.re_input_size)
        out = out.reshape(-1, self.num_neighbor + 1, self.re_input_size)
        out_att = self.attention(out)
        out = out.reshape(-1, 2 * (self.num_neighbor + 1), self.re_input_size)

        return out[:, 0, :], out_att, rules_repr[-1]
