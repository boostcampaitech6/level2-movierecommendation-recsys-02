import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .dataset import MakeSequenceDataSet, BERTRecDataSet, DataLoader
from .model import BERT4Rec
from .inference import inference
from .utils import get_logger, logging_conf, get_expname

logger = get_logger(logger_conf=logging_conf)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, criterion, optimizer, data_loader, device):
    model.train()
    loss_val = 0
    for seq, labels in data_loader:
        logits = model(seq)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1).to(device)

        optimizer.zero_grad()
        loss = criterion(logits, labels)

        loss_val += loss.item()

        loss.backward()
        optimizer.step()
    
    loss_val /= len(data_loader)
    
    return loss_val


def evaluate(model, user_train, user_valid, max_len, bert4rec_dataset, make_sequence_dataset):
    model.eval()

    ndcg = 0.0 # NDCG@10
    hit = 0.0 # HIT@10
    recall = 0.0  # Recall@K

    num_item_sample = 100

    users = [user for user in range(make_sequence_dataset.num_user)]

    for user in users:
        seq = (user_train[user] + [make_sequence_dataset.num_item + 1])[-max_len:]
        rated = user_train[user] + user_valid[user]
        items = user_valid[user] + bert4rec_dataset.random_neg_sampling(rated_item = rated, num_item_sample = num_item_sample)

        with torch.no_grad():
            predictions = -model(np.array([seq]))
            predictions = predictions[0][-1][items] # sampling
            rank = predictions.argsort().argsort()[0].item()
            # ranked_items = predictions.argsort()[:10]

        # if rank < 10: #Top10
        #     ndcg += 1 / np.log2(rank + 2)/10
        #     hit += 1/10
            
        # rank for valid item
        rank_list = predictions.argsort().argsort()[:10].cpu().numpy() 
        recall_cnt = 0
        # ndcg_cnt = 0
        # hit_cnt = 0
        for rank in rank_list:
            if rank < 10:
                ndcg += 1 / np.log2(rank +2)
                hit += 1
                recall_cnt += 1
        recall += recall_cnt/10
        
    ndcg /= len(users)
    hit /= len(users)
    recall /= len(users)
    return ndcg, hit, recall


def trainer(config):
    exp_name = get_expname(config) 
    
    logger.info("Preparing data ...")
    make_sequence_dataset = MakeSequenceDataSet(config = config)
    user_train, user_valid = make_sequence_dataset.get_train_valid_data()
    
    bert4rec_dataset = BERTRecDataSet(
        user_train = user_train,
        max_len = config.max_len,
        num_user = make_sequence_dataset.num_user,
        num_item = make_sequence_dataset.num_item,
        mask_prob = config.mask_prob,
        )
    data_loader = DataLoader(
        bert4rec_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        pin_memory = True,
        num_workers = config.num_workers,
        )
    
    
    logger.info("Building Model ...")
    model = BERT4Rec(
        num_user = make_sequence_dataset.num_user,
        num_item = make_sequence_dataset.num_item,
        hidden_units = config.hidden_units,
        num_heads = config.num_heads,
        num_layers = config.num_layers,
        max_len = config.max_len,
        dropout_rate = config.dropout_rate,
        device = device,
        ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # loss_list = []
    # ndcg_list = []
    # hit_list = []
    
    logger.info(f"Trainning...")
    for epoch in range(1, config.num_epochs + 1):
        logger.info("Start Training: Epoch %s", epoch)
        train_loss = train(
            model = model,
            criterion = criterion,
            optimizer = optimizer,
            data_loader = data_loader,
            device=device)

        ndcg, hit, recall = evaluate(
            model = model,
            user_train = user_train,
            user_valid = user_valid,
            max_len = config.max_len,
            bert4rec_dataset = bert4rec_dataset,
            make_sequence_dataset = make_sequence_dataset,
            )

        logger.info(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f} | RECALL@10: {recall:.5f}')
            
    
    logger.info(f"Inference on Test Data ...")
    inference(model = model,
          user_train = user_train,
          user_valid = user_valid,
          max_len = config.max_len,
          make_sequence_dataset = make_sequence_dataset,
          exp_name = exp_name
          )
    logger.info(f"Inference Finish ...")