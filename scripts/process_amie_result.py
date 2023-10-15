import argparse
import pickle
import os


def process_body_training(s):
    p = s.split()
    q = [i if ("?" in i) else id2rel[int(i)] for i in p]
    ents = [i for i in q if "?" in i]
    deg = {}
    for i in range(len(ents) // 2):
        x = ents[i * 2]
        y = ents[i * 2 + 1]
        deg[x] = deg.get(x, 0) + 1
        deg[y] = deg.get(y, 0) + 1
    for i in deg.keys():
        if i == "?a" or i == "?b":
            assert deg[i] == 1
        else:
            assert deg[i] == 2
    return " ".join(q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="C-05",
        choices=["C-05", "C-10", "C-20", "A-05", "A-10", "A-20"],
        help="The dataset to be used. Choose from 'C-05', 'C-10', 'C-20', 'A-05', 'A-10', 'A-20'.",
    )
    parser.add_argument(
        "--topk", type=int, default=100, help="The number of top rules to use."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="amie-result/",
        help="The directory path containing the amie result files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="The directory path to save the output files.",
    )
    args = parser.parse_args()

    amie = open(os.path.join(args.input_dir, "{}.result".format(args.dataset)), "r").readlines()
    amie = [i for i in amie if "=>" in i]

    Rs = []
    rules = []
    rel2rules = {}
    for line in amie:
        t = line.split("\t")
        conf = float(t[1])
        num = int(t[-4])
        body, head = t[0].split("=>")
        aa, rel, bb = head.split()
        assert aa == "?a" and bb == "?b"
        rel = int(rel)
        rules.append((body, head, conf, num))
        if rel not in rel2rules.keys():
            rel2rules[rel] = []
        rel2rules[rel].append((body, head, conf, num))
        Rs.append(rel)

    if "C" in args.dataset:
        rel2id = {
            "HasPainIntensity": 0,
            "HasFirstSubevent": 1,
            "IsA": 2,
            "NotDesires": 3,
            "CreatedBy": 4,
            "NotCapableOf": 5,
            "LocationOfAction": 6,
            "HasPrerequisite": 7,
            "AtLocation": 8,
            "NotHasProperty": 9,
            "ReceivesAction": 10,
            "LocatedNear": 11,
            "HasPainCharacter": 12,
            "HasSubevent": 13,
            "NotHasA": 14,
            "InstanceOf": 15,
            "HasA": 16,
            "NotMadeOf": 17,
            "RelatedTo": 18,
            "NotIsA": 19,
            "MadeOf": 20,
            "MotivatedByGoal": 21,
            "PartOf": 22,
            "DesireOf": 23,
            "UsedFor": 24,
            "CapableOf": 25,
            "HasProperty": 26,
            "CausesDesire": 27,
            "Desires": 28,
            "Causes": 29,
            "DefinedAs": 30,
            "InheritsFrom": 31,
            "SymbolOf": 32,
            "HasLastSubevent": 33,
        }
    else:
        rel2id = {
            "xReact": 0,
            "xEffect": 1,
            "oReact": 2,
            "oWant": 3,
            "xIntent": 4,
            "xWant": 5,
            "xAttr": 6,
            "xNeed": 7,
            "oEffect": 8,
        }
    id2rel = {}
    for i in rel2id.keys():
        id2rel[rel2id[i]] = i
    print("load {} rules".format(len(rules)))

    used_rel2rules = dict()
    for id_rel in rel2rules.keys():
        rel_rules = rel2rules[id_rel]
        rel_rules.sort(key=lambda x: -x[2])
        used_rel2rules[id_rel] = []
        for idx, line in enumerate(rel_rules):
            ents = [i for i in line[0].split() if "?" in i]
            num_ents = len(list(set(ents)))
            if num_ents < 4:
                continue
            n0 = process_body_training(line[0])
            n1 = process_body_training(line[1])
            if idx < 100 and float(line[3]) > 400:
                aa = n0.split()
                L = [aa[1], aa[4], aa[7], n1.split()[1]]
                A = aa[0:3]
                B = aa[3:6]
                C = aa[6:9]
                if "?a" in B:
                    A, B = B, A
                if "?a" in C:
                    A, C = C, A
                if "?b" in C:
                    B, C = C, B
            used_rel2rules[id_rel].append([line[0], line[1], line[2], line[3]])
            assert num_ents == 4
            if len(used_rel2rules[id_rel]) == args.topk:
                break

    used_id2rules = dict()
    for i in used_rel2rules.keys():
        used_id2rules[id2rel[i]] = used_rel2rules[i]

    for i in used_id2rules.keys():
        print("{}\t{}".format(i, len(used_id2rules[i])))
    with open(os.path.join(args.output_dir, "{}-rules-top-{}.pkl".format(args.dataset, args.topk)), "wb") as f:
        pickle.dump(used_id2rules, f)
    print("DONE")
