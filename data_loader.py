import logging
import os
import pickle
import random

import numpy as np
from sentence_transformers import SentenceTransformer


class MyDict:
    def __init__(self, name) -> None:
        self.name = name
        self.k2v = dict()
        self.v2k = dict()
        self.cnt = 0

    def __getitem__(self, key) -> int:
        if type(key) == str:
            if key in self.k2v.keys():
                return self.k2v[key]
            else:
                self.k2v[key] = self.cnt
                self.v2k[self.cnt] = key
                self.cnt += 1
                return self.k2v[key]
        else:
            return self.v2k[key]

    def dump(self, file):
        for i in range(self.cnt):
            print("{}\t{}".format(i, self.v2k[i]), file=file)


class TripleDict:
    def __init__(self) -> None:
        self.triple_set = set()
        self.triple_list = []
        self.h2rt = dict()
        self.t2rh = dict()
        self.r2h = dict()
        self.r2t = dict()
        self.r2ht = dict()
        self.d_in = dict()
        self.d_out = dict()
        self.cnt = 0

    def __getitem__(self, key):
        return self.triple_list[key]

    def add(self, h, r, t):
        if not (h, r, t) in self.triple_set:
            self.triple_list.append((h, r, t))
            self.triple_set.add((h, r, t))
            self.cnt += 1
        if not h in self.h2rt.keys():
            self.h2rt[h] = {(r, t)}
        else:
            self.h2rt[h].add((r, t))
        if not t in self.t2rh.keys():
            self.t2rh[t] = {(r, h)}
        else:
            self.t2rh[t].add((r, h))
        if not r in self.r2h.keys():
            self.r2h[r] = {h}
            self.r2t[r] = {t}
        else:
            self.r2h[r].add(h)
            self.r2t[r].add(t)
        if not r in self.r2ht.keys():
            self.r2ht[r] = {(h, r, t)}
        else:
            self.r2ht[r].add((h, r, t))
        if not h in self.d_out.keys():
            self.d_out[h] = dict()
        if not r in self.d_out[h].keys():
            self.d_out[h][r] = set()
        self.d_out[h][r].add(t)

        if not t in self.d_in.keys():
            self.d_in[t] = dict()
        if not r in self.d_in[t].keys():
            self.d_in[t][r] = set()
        self.d_in[t][r].add(h)

    def valid(self, h, r, t):
        return (h, r, t) in self.triple_set

    def to_list(self):
        return self.triple_list

    def get_neighbors(self, e):
        neighbors = []
        ns = [e]
        for ee in ns:
            if ee in self.h2rt.keys():
                for r, t in list(self.h2rt[ee]):
                    neighbors.append((ee, r, t))
            if ee in self.t2rh.keys():
                for r, h in self.t2rh[ee]:
                    neighbors.append((h, r, ee))
        return neighbors

    def get_rel_neighbors(self, r):
        return list(self.r2ht[r])


class Dataset:
    def __init__(self, config) -> None:
        self.config = config
        self.entity = MyDict("entity")
        self.relation = MyDict("relation")
        self.triple = TripleDict()

    def read(self):
        logging.info(
            "start reading {} dataset at {}".format(
                self.config.dataset_name, self.config.dataset_path
            )
        )
        datas = []
        for file_name in ["train.txt", "valid.txt", "test.txt"]:
            with open(os.path.join(self.config.dataset_path, file_name), "r") as f:
                datas.extend(f.readlines())

        logging.info("read {} lines".format(len(datas)))

        for line in datas:
            sep = line.strip("\n").split("\t")
            assert len(sep) == 3
            if self.config.dataset_name == "conceptnet":
                r, h, t = sep
            elif self.config.dataset_name == "atomic":
                h, r, t = sep
            else:
                h, r, t = sep
                # raise ValueError

            h = self.entity[h]
            r = self.relation[r]
            t = self.entity[t]
            self.triple.add(h, r, t)

        logging.info(
            "#entity: {}, #relation: {}, #triple: {}".format(
                self.entity.cnt, self.relation.cnt, self.triple.cnt
            )
        )

        self.read_rules()

        return self.get_train_list_size()

    def gen_neg_sample(self, train_list, k=1):
        logging.info("generate {} NEGATIVE SAMPLES per positive".format(k))
        neg_train_list = []
        for hrt, label in train_list:
            neg_list = []
            for l in range(k):
                h, r, t = hrt
                if random.randint(0, 1) == 0:
                    h = random.randint(0, self.entity.cnt - 1)
                    while self.triple.valid(h, r, t):
                        h = random.randint(0, self.entity.cnt - 1)
                else:
                    t = random.randint(0, self.entity.cnt - 1)
                    while self.triple.valid(h, r, t):
                        t = random.randint(0, self.entity.cnt - 1)
                neg_list.append((h, r, t))
            neg_train_list.append(neg_list)
        return neg_train_list

    def load_error_triplets(self):
        with open(
            os.path.join(
                self.config.dataset_path,
                "errors/{}-error.txt".format(self.config.dataset),
            ),
            "r",
        ) as f:
            raw_err_list = f.readlines()
        logging.info("read {} lines".format(len(raw_err_list)))
        err_list = []
        for line in raw_err_list:
            sep = line.strip("\n").split("\t")
            assert len(sep) == 3
            h, r, t = sep

            h = self.entity.k2v[h]
            r = self.relation.k2v[r]
            t = self.entity.k2v[t]
            err_list.append((h, r, t))
        return err_list

    def get_train_list_size(self):
        err_list = self.load_error_triplets()
        nr_error = len(err_list)
        logging.info("load {} error triples".format(nr_error))
        self.train_list = [(i, 0) for i in self.triple.to_list()] + [
            (i, 1) for i in err_list
        ]
        random.shuffle(self.train_list)
        self.neg_list = self.gen_neg_sample(self.train_list, k=self.config.neg_cnt)
        return len(self.train_list), nr_error

    def gen_train_batch(self, idx_list):
        triple_list = []
        masks = []
        for idx in idx_list:
            triple_list.append(self.train_list[idx][0])
            for j in range(self.config.neg_cnt):
                triple_list.append(self.neg_list[idx][j])
            masks.append(1)
        batch_list = []
        batch_instances = []
        for h, r, t in triple_list:
            h_n = self.triple.get_neighbors(h)
            t_n = self.triple.get_neighbors(t)
            if len(h_n) > self.config.num_neighbor:
                hh_neighbors = random.sample(h_n, k=self.config.num_neighbor)
            else:
                hh_neighbors = random.choices(h_n, k=self.config.num_neighbor)
            if len(t_n) > self.config.num_neighbor:
                tt_neighbors = random.sample(t_n, k=self.config.num_neighbor)
            else:
                tt_neighbors = random.choices(t_n, k=self.config.num_neighbor)
            batch_list = (
                batch_list + [(h, r, t)] + hh_neighbors + [(h, r, t)] + tt_neighbors
            )
            batch_instances = batch_instances + self.sample_rule_instances(h, r, t)

        batch_rules_h = []
        batch_rules_r = []
        batch_rules_t = []
        batch_rules_c = []
        for inst in batch_instances:
            A, B, C, conf = inst
            batch_rules_h.extend([A[0], B[0], C[0]])
            batch_rules_r.extend([A[1], B[1], C[1]])
            batch_rules_t.extend([A[2], B[2], C[2]])
            batch_rules_c.append(conf)

        batch_h = [x[0] for x in batch_list]
        batch_r = [x[1] for x in batch_list]
        batch_t = [x[2] for x in batch_list]
        return (
            batch_h,
            batch_r,
            batch_t,
            batch_rules_h,
            batch_rules_r,
            batch_rules_t,
            batch_rules_c,
            masks,
            len(idx_list),
        )

    def gen_test_batch(self, idx_list):
        triple_list = []
        labels = []
        masks = []
        for idx in idx_list:
            triple_list.append(self.train_list[idx][0])
            labels.append(self.train_list[idx][1])
            masks.append(1)
        batch_list = []
        batch_instances = []
        for h, r, t in triple_list:
            h_n = self.triple.get_neighbors(h)
            t_n = self.triple.get_neighbors(t)
            if len(h_n) > self.config.num_neighbor:
                hh_neighbors = random.sample(h_n, k=self.config.num_neighbor)
            else:
                hh_neighbors = random.choices(h_n, k=self.config.num_neighbor)
            if len(t_n) > self.config.num_neighbor:
                tt_neighbors = random.sample(t_n, k=self.config.num_neighbor)
            else:
                tt_neighbors = random.choices(t_n, k=self.config.num_neighbor)
            batch_list = (
                batch_list + [(h, r, t)] + hh_neighbors + [(h, r, t)] + tt_neighbors
            )
            batch_instances = batch_instances + self.sample_rule_instances(h, r, t)
        batch_rules_h = []
        batch_rules_r = []
        batch_rules_t = []
        batch_rules_c = []
        for inst in batch_instances:
            A, B, C, conf = inst
            batch_rules_h.extend([A[0], B[0], C[0]])
            batch_rules_r.extend([A[1], B[1], C[1]])
            batch_rules_t.extend([A[2], B[2], C[2]])
            batch_rules_c.append(conf)

        batch_h = [x[0] for x in batch_list]
        batch_r = [x[1] for x in batch_list]
        batch_t = [x[2] for x in batch_list]
        return (
            batch_h,
            batch_r,
            batch_t,
            batch_rules_h,
            batch_rules_r,
            batch_rules_t,
            batch_rules_c,
            labels,
            idx_list,
            masks,
            len(idx_list),
        )

    def get_pretrained_embedding(self):
        ent_ptlm_embed = np.zeros((self.entity.cnt, self.config.ptlm_embedding_dim))
        rel_ptlm_embed = np.zeros((self.relation.cnt, self.config.ptlm_embedding_dim))

        embed_file_name = self.config.ptlm_model.replace("/", "_")

        ent_embed_file = (
            "./" + embed_file_name + "-" + self.config.dataset_name + "-entity"
        )
        rel_embed_file = (
            "./" + embed_file_name + "-" + self.config.dataset_name + "-relation"
        )
        if not os.path.exists(ent_embed_file + ".npy") or not os.path.exists(
            rel_embed_file + ".npy"
        ):
            model = SentenceTransformer(self.config.ptlm_model)
        if os.path.exists(ent_embed_file + ".npy"):
            logging.info(
                "entity embedding file {} found, start loading ...".format(
                    ent_embed_file
                )
            )
            ent_ptlm_embed = np.load(ent_embed_file + ".npy")
            logging.info("loaded!")
        else:
            logging.info("start generating entity embeddings ...")
            ss = [self.entity[i] for i in range(self.entity.cnt)]
            emb = model.encode(ss)
            ent_ptlm_embed[:, :] = emb
            logging.info("entity embedding generated, start saving ...")
            np.save(ent_embed_file, ent_ptlm_embed)
            logging.info("saved!")

        if os.path.exists(rel_embed_file + ".npy"):
            logging.info(
                "relation embedding file {} found, start loading ...".format(
                    rel_embed_file
                )
            )
            rel_ptlm_embed = np.load(rel_embed_file + ".npy")
            logging.info("loaded!")
        else:
            logging.info("start generating relation embeddings ...")
            ss = [self.relation[i] for i in range(self.relation.cnt)]
            emb = model.encode(ss)
            rel_ptlm_embed[:, :] = emb
            logging.info("relation embedding generated, start saving ...")
            np.save(rel_embed_file, rel_ptlm_embed)
            logging.info("saved!")

        logging.info("generating PTLM embeddings DONE!!!")
        return ent_ptlm_embed, rel_ptlm_embed

    def read_rules(self):
        logging.info("loading rules ...")
        with open(
            os.path.join(
                self.config.dataset_path,
                "rules/{}-rules-top-{}.pkl".format(
                    self.config.dataset, self.config.rule_top_k
                ),
            ),
            "rb",
        ) as f:
            self.rel2rules = pickle.load(f)
        for i in self.rel2rules:
            logging.info("    {} => {} rules".format(i, len((self.rel2rules[i]))))
        logging.info("rules loaded")

    def sample_rule_instances(self, h, r, t) -> list:
        rule_set = self.rel2rules.get(self.relation[r], {})

        def f_in(x, r):
            if not x in self.triple.d_in.keys():
                return {}
            if not r in self.triple.d_in[x].keys():
                return {}
            return self.triple.d_in[x][r]

        def f_out(x, r):
            if not x in self.triple.d_out.keys():
                return {}
            if not r in self.triple.d_out[x].keys():
                return {}
            return self.triple.d_out[x][r]

        instances = []
        for rule in rule_set:
            body = rule[0].split()
            conf = float(rule[2])
            A, B, C = body[:3], body[3:6], body[6:]
            ar = self.relation[A[1]]
            br = self.relation[B[1]]
            cr = self.relation[C[1]]
            if "?a" in B:
                A, B = B, A
            if "?a" in C:
                A, C = C, A
            if "?b" in C:
                B, C = C, B
            # make sure: 1)"?a" in A. 2)"?b" in B.

            X = f_out(h, ar) if "?a" == A[0] else f_in(h, ar)
            Y = f_out(t, br) if "?b" == B[0] else f_in(t, br)

            if C[0] in A:
                for u in X:
                    for v in f_out(u, cr):
                        if v in Y:
                            # [hu, vt, uv]
                            inst = []
                            if "?a" == A[0]:
                                inst.append([h, ar, u])
                            else:
                                inst.append([u, ar, h])
                            inst.append([u, cr, v])
                            if "?b" == B[0]:
                                inst.append([t, br, v])
                            else:
                                inst.append([v, br, t])
                            inst.append(1)
                            instances.append(inst)
            else:
                for u in Y:
                    for v in f_out(u, cr):
                        if v in X:
                            # [hv, ut, uv]
                            inst = []
                            if "?a" == A[0]:
                                inst.append([h, ar, v])
                            else:
                                inst.append([v, ar, h])
                            inst.append([u, cr, v])
                            if "?b" == B[0]:
                                inst.append([t, br, u])
                            else:
                                inst.append([u, br, t])
                            inst.append(1)
                            instances.append(inst)

        if len(instances) >= self.config.rule_inst_cnt:
            instances = random.sample(instances, k=self.config.rule_inst_cnt)
        elif len(instances) > 0:
            instances = random.choices(instances, k=self.config.rule_inst_cnt)
        else:
            instances = [
                [
                    [h, r, t],
                ]
                * 3
                + [
                    1,
                ],
            ] * self.config.rule_inst_cnt
        assert len(instances) == self.config.rule_inst_cnt

        return instances
