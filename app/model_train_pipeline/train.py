import numpy as np
import torch
from torch import cuda
from torch import nn
from model import *
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from evaluate import evaluate
import argparse

def train(model, dataset, eval_size=0, num_epochs=15, model_name='modelname'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    train_losses = []
    eval_losses = []

    X_train, X_test, y_train, y_test = train_test_split(dataset[['prompt', 'full_address']], dataset['scores'],
                                                        test_size=eval_size, random_state=42)

    for epoch in range(num_epochs):

        if epoch % 1 == 0 and X_test is not None:
            model.eval()
            eval_loss = evaluate(model, X_test, y_test, criterion)

        model.train()
        train_epoch_losses = []

        for index in range(X_train.shape[0]):
            text = X_train.iloc[index]['prompt']
            target = X_train.iloc[index]['full_address']
            score = torch.tensor(y_train.iloc[index], dtype=torch.float32).to(model.device)

            output = model(text, target).squeeze()

            optimizer.zero_grad()

            loss = criterion(output, score).to(model.device)

            loss.backward()
            optimizer.step()

            train_epoch_losses.append(loss.item())

        train_loss = np.mean(train_epoch_losses)

        train_losses.append(train_loss)

        torch.save(model.state_dict(),
                   f"../additional_data/{model_name}_{epoch}epoch.pt")

    return model

def __main__():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest='data_path', type=str, default= './neg_40k-2.csv')
    args = parser.parse_args()

    model = BertScorer()
    dataset = pd.read_csv(args['data_path'])

    train(model, dataset)
