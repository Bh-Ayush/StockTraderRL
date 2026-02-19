# Stock Trading with Reinforcement Learning

A reinforcement learning system that trains DQN, PPO, and A2C agents to trade stocks using technical indicators as observations. Compares agent performance across multiple stock symbols.

## Project Structure

```
stock-rl-trading/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Data download, technical indicators, stock universe
│   ├── environments.py        # Gymnasium environments (training + testing)
│   └── reward_models.py       # LSTM, Linear NN, and LLM reward predictors
├── notebooks/
│   └── main.ipynb             # Full pipeline: train, test, visualize
└── models/                    # Saved model weights (gitignored)
```

## Setup

```bash
# Clone the repo
git clone https://github.com/<you>/stock-rl-trading.git
cd stock-rl-trading

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Open and run the notebook:

```bash
cd notebooks
jupyter notebook main.ipynb
```

The notebook walks through:

1. **Data preprocessing** — Downloads OHLCV data via yfinance and computes technical indicators (SMA, RSI, OBV, ATR, CCI)
2. **Reward model training** — Trains LSTM and feedforward NN models to predict next-day close prices
3. **RL agent training** — Trains DQN, PPO, and A2C agents using stable-baselines3
4. **Evaluation** — Tests all three agents on held-out symbols and compares portfolio values and cumulative rewards
5. **Real-time data** — Optional live price tracking and news feed

## Key Components

### Environment

The `StockTradingEnv` (gymnasium-compatible) provides:
- **Actions**: Buy (0), Sell (1), Hold (2)
- **Observations**: 10-day sliding window × 10 features (OHLCV + 5 technical indicators)
- **Reward**: Configurable — statistical (default), LSTM-based, NN-based, or LLM-based

### Stock Universe

35 stocks across 7 NASDAQ-100 sectors: Consumer Discretionary, Consumer Staples, Health Care, Industrial, Technology, Telecommunications, Utilities.

### Reward Functions

| Type | Description |
|------|-------------|
| Statistical | Rule-based using RSI, SMA, CCI signals |
| LSTM | Learned next-day-close predictor (sequence model) |
| Linear NN | Feedforward next-day-close predictor |
| LLM | GPT-3.5 as reward evaluator (requires OpenAI API key) |

## Optional: LLM Reward

To use the LLM-based reward function, set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Tech Stack

- **RL**: stable-baselines3 (DQN, PPO, A2C)
- **Environment**: gymnasium
- **Data**: yfinance, ta (technical analysis)
- **Models**: PyTorch (LSTM, feedforward NN)
- **Visualization**: matplotlib, tensorboard
