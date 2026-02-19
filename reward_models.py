"""
Reward prediction models for the stock trading RL environment.

Contains:
    - LSTMRewardPredictor: LSTM-based next-day-close predictor
    - LinearRewardPredictor: feedforward NN next-day-close predictor
    - Training utilities
    - LLM-based reward function (requires OpenAI API key)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import add_technical_indicators, download_stock, ALL_STOCKS


# ──────────────────────────────────────────────────────────────────────────────
# LSTM Reward Predictor
# ──────────────────────────────────────────────────────────────────────────────
class LSTMRewardPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int = 64, output_size: int = 1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(1, x.size(0), self.hidden_layer_size, device=x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_layer_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.linear(out[:, -1, :])


# ──────────────────────────────────────────────────────────────────────────────
# Linear (feedforward) Reward Predictor
# ──────────────────────────────────────────────────────────────────────────────
class LinearRewardPredictor(nn.Module):
    def __init__(self, input_size: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100,
    print_every: int = 25,
) -> list[float]:
    """Generic training loop. Returns per-epoch average losses."""
    model.train()
    losses: list[float] = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # targets may be (B,) or (B,1); ensure matching shape
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if epoch % print_every == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}  Loss: {avg:.4f}")
    return losses


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation utilities
# ──────────────────────────────────────────────────────────────────────────────
def prepare_linear_data(
    symbols: list[str],
    start_date: str = "2020-01-01",
    end_date: str = "2021-01-01",
    batch_size: int = 32,
    test_size: float = 0.2,
):
    """Download data for multiple symbols, compute features, return DataLoaders + scaler."""
    all_data = pd.DataFrame()
    for sym in symbols:
        try:
            sd = download_stock(sym, start_date, end_date)
            all_data = pd.concat([all_data, sd], axis=0)
        except Exception as e:
            print(f"Skipping {sym}: {e}")

    features = all_data[["Close", "SMA", "RSI", "OBV", "ATR_14", "CCI_20"]].values
    targets = all_data["NextDayClose"].values

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=test_size, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_s, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size),
        scaler,
    )


def create_sequences(data: np.ndarray, seq_len: int):
    """Sliding-window sequences for LSTM. Last column = target."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len, :-1])
        y.append(data[i + seq_len - 1, -1])
    return np.array(X), np.array(y)


def prepare_lstm_data(
    symbols: list[str],
    start_date: str = "2020-01-01",
    end_date: str = "2021-01-01",
    sequence_length: int = 10,
    batch_size: int = 32,
    test_size: float = 0.2,
):
    """Prepare sequenced data for LSTM training."""
    all_data = pd.DataFrame()
    for sym in symbols:
        try:
            sd = download_stock(sym, start_date, end_date)
            all_data = pd.concat([all_data, sd], axis=0)
        except Exception as e:
            print(f"Skipping {sym}: {e}")

    cols = ["Close", "SMA", "RSI", "OBV", "ATR_14", "CCI_20", "NextDayClose"]
    values = all_data[cols].values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)

    X, y = create_sequences(scaled, sequence_length)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.float32),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size),
        scaler,
    )


# ──────────────────────────────────────────────────────────────────────────────
# LLM-based reward (requires OPENAI_API_KEY env variable)
# ──────────────────────────────────────────────────────────────────────────────
_ENV_DESCRIPTION = """
class StockEnv(gym.Env):
    action_space = gym.spaces.Discrete(3)  # Buy(0), Sell(1), Hold(2)
    observation_space: 10 features per timestep:
        ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'OBV', 'ATR_14', 'CCI_20']
    initial_balance = 10000
"""


def get_llm_reward(action: int, obs: list, client=None) -> str:
    """
    Query an OpenAI model to produce a numerical reward.

    Requires either:
        - passing an openai.OpenAI client, or
        - setting the OPENAI_API_KEY environment variable
    """
    if client is None:
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY from env

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that returns a single numerical reward. "
                    "Given the following RL trading environment:\n" + _ENV_DESCRIPTION
                ),
            },
            {
                "role": "user",
                "content": (
                    "Act as the calculate_reward function. It takes an action "
                    "Buy(0), Sell(1), Hold(2) and 10 observations "
                    "['Open','High','Low','Close','Volume','SMA','RSI','OBV','ATR_14','CCI_20']. "
                    "Return a single number."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I will evaluate the action against the observation and "
                    "return a single numerical reward."
                ),
            },
            {
                "role": "user",
                "content": f"Action {action}, observation {obs}. What is the reward?",
            },
        ],
    )
    return response.choices[0].message.content
