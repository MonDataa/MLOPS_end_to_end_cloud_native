import ray
from ray import tune
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def build_synthetic():
    X = np.random.rand(100, 2).astype('float32')
    y = (X[:, 0] + X[:, 1]).astype('float32')
    return X, y


def model_fn(config):
    return nn.Sequential(
        nn.Linear(2, config.get('hidden_units', 16)),
        nn.ReLU(),
        nn.Linear(config.get('hidden_units', 16), 1),
    )


def train(config):
    model = model_fn(config)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.MSELoss()
    X, y = build_synthetic()
    X = torch.from_numpy(X)
    y = torch.from_numpy(y).unsqueeze(1)
    for epoch in range(5):
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        tune.report(loss=loss.item())


def main():
    ray.init(ignore_reinit_error=True)
    analysis = tune.run(
        train,
        config={
            'lr': tune.grid_search([0.01, 0.005, 0.001]),
            'hidden_units': tune.grid_search([8, 16]),
        },
        metric='loss',
        mode='min',
    )
    print('Best config:', analysis.best_config)


if __name__ == '__main__':
    main()
