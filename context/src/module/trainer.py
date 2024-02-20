import os
import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import roc_auc_score
from .dataset import MakeDataset, RatingDataset
from .model import FactorizationMachine, FieldAwareFactorizationMachineModel, DeepFM
from .inference import inference
from .utils import get_logger, logging_conf, get_expname

logger = get_logger(logger_conf=logging_conf)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, data_loader, criterion, optimizer, device):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    train_loss = 0
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y.float())
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if batch % 1000 == 0:
            loss = loss.item()
            current = batch * len(X)
            logger.info(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    
    return train_loss


def test(model, data_loader, criterion, device):
    num_batches = len(data_loader)
    test_loss, y_all, pred_all = 0, list(), list()

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y.float()).item() / num_batches
            y_all.append(y)
            pred_all.append(pred)

    y_all = torch.cat(y_all).cpu()
    pred_all = torch.cat(pred_all).cpu()

    auc = roc_auc_score(y_all, pred_all).item()
    logger.info(f"Test Error: \n  AUC: {auc:>8f} \n  Avg loss: {test_loss:>8f}")

    return auc, test_loss


def trainer(args):
    exp_name = get_expname(args)
    if args.wandb:   
        # wandb initialization
        wandb.init(entity='raise_level2', project="Movie_Rec")
        wandb.run.name = exp_name
        wandb.run.save() 
    
    logger.info("Preparing data ...")
    # Example usage:
    rating_data_path = os.path.join(args.data_path, "train_ratings.csv")
    genre_data_path = os.path.join(args.data_path, "genres.tsv")
    
    num_negative_samples = args.num_nag_samples

    dataset = MakeDataset(rating_data_path, genre_data_path, num_negative=num_negative_samples)
    raw_rating_df, user_group_dfs, raw_genre_df = dataset.preprocess()

    n_data, n_user, n_item, n_genre, users_dict, items_dict, genre_dict = dataset.get_statistics()
    
    # Access preprocessed data
    data = dataset.data

    logger.info("Preparing dataset ...")
    user_col = torch.tensor(data.loc[:,'user'])
    item_col = torch.tensor(data.loc[:,'item'])
    genre_col = torch.tensor(data.loc[:,'genre'])

    X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1)], dim=1)
    y = torch.tensor(list(data.loc[:,'rating']))

    dataset = RatingDataset(X, y)
    train_ratio = 0.9

    train_size = int(train_ratio * len(data))
    test_size = len(data) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    logger.info("Building Model ...")
    if args.model == 'fm':
        input_dim = data.shape[1] - 1
        model = FactorizationMachine(input_dim, args.embed_dim).to(device)
    elif args.model == 'ffm':
        ffm_field_dims = np.array([len(users_dict), len(items_dict), len(genre_dict)]) #, dtype=np.uint32)
        model = FieldAwareFactorizationMachineModel(ffm_field_dims, args.embed_dim).to(device)
    elif args.model == 'deepfm':
        input_dims = [n_user, n_item, n_genre]
        model = DeepFM(input_dims, args.embed_dim, args.mlp_dims, drop_rate=0.01).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    
    
    logger.info("Trainning...")
    for t in range(args.epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        auc, val_loss = test(model, test_loader, criterion, device)

        if args.wandb:
            wandb.log(dict(train_loss=train_loss,
                        val_loss=val_loss,
                        auc=auc))
    
            
    
    logger.info(f"Inference on Test Data ...")
    submit_df = inference(model, items_dict, users_dict, raw_genre_df, user_group_dfs, n_item, device)
    submit_df.to_csv(f"outputs/{exp_name}_submission.csv", index=False)
    logger.info(f"Inference Finish ...")
    
    if args.wandb:
        submission_artifact = wandb.Artifact(f'{exp_name}_submission', type='output')
        submission_artifact.add_file(local_path=f"outputs/{exp_name}_submission.csv")
        wandb.log_artifact(submission_artifact)
        wandb.finish()
