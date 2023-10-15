class Config:
    dataset = ""
    dataset_name = ""
    dataset_path = ""
    error_ratio = 0
    total_ent = 0
    total_rel = 0
    total_triple = 0
    ptlm_model = None
    nr_train_samples = 0
    nr_error = 0
    output_tsv = False

    hidden_size = 100
    num_layers = 1
    dropout = 0.2
    rnn_input_size = 100
    rnn_hidden_size = 100
    rnn_num_layers = 2
    num_neighbor = 31
    ptlm_embedding_dim = 768
    model_name = ""
    neg_cnt = 1
    rule_inst_cnt = 21
    rule_top_k = 100

    learning_rate = 1e-3
    batch_size = 256
    num_epochs = 10
    use_ptlm = True

    alpha = 0.2
    gama = 0.5
    local_lambda = 0.1
    global_lambda = 0.1

    device = "cuda"
    seed = 5
