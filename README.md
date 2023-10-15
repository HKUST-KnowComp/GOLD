# GOLD: A Global and Local-aware Denoising Framework for Commonsense Knowledge Graph Noise Detection
This is the official code and data repository for the EMNLP2023 Findings paper: [Gold: A Global and Local-aware Denoising Framework for Commonsense Knowledge Graph Noise Detection]().

![avatar](demo/featured.jpg)

## Environment

Install the project dependencies
```shell
pip install -r requirements.txt
```

## Dataset
The ConceptNet and Atomic datasets are already provided in the `dataset/` directory. Following previous work[1, 2], we generated 5%, 10%, and 20% noisy triples for both datasets, and these files are stored in `errors/` subfolders. Additionally, we applied AMIE[3] on the noisy CSKGs to mine a set of top-ranked rules, which are stored in `rules/` subfolders.



## Model Training

To train the model, you can use the following command

```shell
python gold.py \
--dataset C-05 \
--model_name train \
--epoch 10 \
--batch_size 256 \
--topk 100 \
--ptlm_model sentence-transformers/sentence-t5-xxl \
--lr 0.001 \
--local_lambda 0.1 \
--global_lambda 0.01 \
--neg_cnt 1 \
--seed 5 \
--output_tsv
```

## Scripts for Rule Mining

To mine rules from a knowledge base, please refer to [AMIE](https://github.com/dig-team/amie)[3] and follow the instructions provided to generate rules. The resulting files obtained from mining on ConceptNet and Atomic can be found in the `scripts/amie-result` directory for reference. Then, you can use `scripts/process_amie_result.py` to process the rules from the CSKG and keep the `topk` rules for each relation. Here is an example of the code:

```shell
python process_amie_result.py --dataset C-05 --topk 500
```


## Reference

[1] "Does william shakespeare REALLY write hamlet? knowledge representation learning with confidence". https://github.com/thunlp/CKRL.git

[2] "Contrastive Knowledge Graph Error Detection". https://github.com/DEEP-PolyU/CAGED_CIKM22.git

[3] "Association Rule Mining under Incomplete Evidence". https://github.com/dig-team/amie.git