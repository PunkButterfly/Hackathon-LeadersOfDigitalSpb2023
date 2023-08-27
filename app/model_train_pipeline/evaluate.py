import numpy as np
import pandas as pd
import torch


def evaluate(model, X_test, y_test, criterion):
    eval_losses = []

    for index in range(X_test.shape[0]):
        text = X_test.iloc[index]['prompt']
        target = X_test.iloc[index]['full_address']
        score = torch.tensor(y_test.iloc[index], dtype=torch.float32).to(model.device)

        with torch.no_grad():
            output = model(text, target).squeeze()

        loss = criterion(output, score).to(model.device)

        eval_losses.append(loss.item())

    eval_loss = np.mean(eval_losses)
    print(f'Eval Loss: {eval_loss}')
    return eval_loss
