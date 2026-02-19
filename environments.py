"""
Gymnasium environments for stock trading with RL agents.

Two environments:
    - StockTradingEnv: training environment with configurable reward functions
    - StockTestEnv: testing environment that evaluates on a specific symbol
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.preprocessing import add_technical_indicators, download_stock


# ──────────────────────────────────────────────────────────────────────────────
# Training environment
# ──────────────────────────────────────────────────────────────────────────────
class StockTradingEnv(gym.Env):
    """
    A stock trading environment for reinforcement learning.

    Actions:
        0 = Buy one share
        1 = Sell one share
        2 = Hold

    Observation:
        A (window_size × 10) matrix of OHLCV + technical indicators.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        stock_symbols: list[str],
        window_size: int,
        start_date: str,
        end_date: str,
        reward_fn=None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.stock_symbols = stock_symbols
        self.window_size = window_size
        self.start_date = start_date
        self.end_date = end_date
        self.render_mode = render_mode
        self.reward_fn = reward_fn or self._default_reward

        # Spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(window_size, 10), dtype=np.float64
        )

        # Portfolio state
        self.initial_balance = 10_000
        self.current_balance = self.initial_balance
        self.shares_held = 0
        self.current_portfolio_value = self.current_balance

        # Will be populated on reset()
        self.current_symbol = None
        self.stock_data = None
        self.current_step = window_size

        # Tracking for render
        self._last_action = None
        self._last_reward = None
        self.portfolio_history: list[float] = []

        # Download initial data so SB3 can inspect the env
        self._load_random_stock()

    # ── data helpers ──────────────────────────────────────────────────────
    def _load_random_stock(self):
        self.current_symbol = np.random.choice(self.stock_symbols)
        self.stock_data = download_stock(
            self.current_symbol, self.start_date, self.end_date
        )

    # ── observation ───────────────────────────────────────────────────────
    def _next_observation(self) -> np.ndarray:
        frame = self.stock_data.iloc[
            self.current_step - self.window_size : self.current_step
        ]
        return frame[
            ["Open", "High", "Low", "Close", "Volume", "SMA", "RSI", "OBV", "ATR_14", "CCI_20"]
        ].values

    # ── default reward ────────────────────────────────────────────────────
    @staticmethod
    def _default_reward(env, action: int) -> float:
        """Statistical reward based on technical indicator signals."""
        obs_row = env.stock_data.iloc[env.current_step]
        sma = obs_row["SMA"]
        close = obs_row["Close"]
        rsi = obs_row["RSI"]
        cci = obs_row["CCI_20"]
        reward = 0.0

        price_above_sma = close > sma
        price_below_sma = close < sma
        oversold = rsi < 30
        overbought = rsi > 70
        cci_oversold = cci < -100
        cci_overbought = cci > 100

        if action == 0:  # Buy
            if price_above_sma:
                reward += 2
            if oversold:
                reward += 2
            if cci_oversold:
                reward += 2
            if price_above_sma and oversold and cci_oversold:
                reward += 5
        elif action == 1:  # Sell
            if price_below_sma:
                reward += 2
            if overbought:
                reward += 2
            if cci_overbought:
                reward += 2
            if price_below_sma and overbought and cci_overbought:
                reward += 5

        return reward * reward

    # ── gym interface ─────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_random_stock()
        self.current_step = self.window_size
        self.current_balance = self.initial_balance
        self.shares_held = 0
        self.current_portfolio_value = self.current_balance
        self.portfolio_history = []
        self._last_action = None
        self._last_reward = None
        return self._next_observation(), {}

    def step(self, action: int):
        current_price = self.stock_data.iloc[self.current_step]["Close"]
        invalid = False

        if action == 0:  # Buy
            if self.current_balance >= current_price:
                self.shares_held += 1
                self.current_balance -= current_price
            else:
                invalid = True
        elif action == 1:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.current_balance += current_price
            else:
                invalid = True

        self.current_portfolio_value = (
            self.current_balance + self.shares_held * current_price
        )
        self.portfolio_history.append(self.current_portfolio_value)

        if invalid:
            reward = -10.0
        else:
            reward = float(self.reward_fn(self, action))

        self.current_step += 1
        terminated = self.current_step >= len(self.stock_data) - 1
        truncated = False

        self._last_action = action
        self._last_reward = reward

        return self._next_observation(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human" and self._last_action is not None:
            action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
            print(
                f"Step: {self.current_step} | "
                f"Action: {action_map.get(self._last_action, '?')} | "
                f"Reward: {self._last_reward:.2f} | "
                f"Balance: ${self.current_balance:,.2f} | "
                f"Shares: {self.shares_held} | "
                f"Portfolio: ${self.current_portfolio_value:,.2f}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Testing environment (specific symbol, simpler reward)
# ──────────────────────────────────────────────────────────────────────────────
class StockTestEnv(gym.Env):
    """
    Environment for evaluating trained agents on a specific stock symbol.
    Reward = portfolio_value − initial_balance  (pure P&L signal).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        stock_symbols: list[str],
        window_size: int,
        start_date: str,
        end_date: str,
        render_mode: str | None = None,
        specific_symbol: str | None = None,
    ):
        super().__init__()
        self.stock_symbols = stock_symbols
        self.window_size = window_size
        self.start_date = start_date
        self.end_date = end_date
        self.render_mode = render_mode
        self.specific_symbol = specific_symbol

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(window_size, 10), dtype=np.float64
        )

        self.initial_balance = 10_000
        self.current_balance = self.initial_balance
        self.shares_held = 0
        self.current_portfolio_value = self.current_balance
        self.current_symbol = specific_symbol or np.random.choice(stock_symbols)
        self.stock_data = download_stock(
            self.current_symbol, self.start_date, self.end_date
        )
        self.current_step = window_size

        self._last_action = None
        self.portfolio_history: list[float] = []

    def _next_observation(self) -> np.ndarray:
        frame = self.stock_data.iloc[
            self.current_step - self.window_size : self.current_step
        ]
        return frame[
            ["Open", "High", "Low", "Close", "Volume", "SMA", "RSI", "OBV", "ATR_14", "CCI_20"]
        ].values

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.specific_symbol:
            self.current_symbol = np.random.choice(self.stock_symbols)
        self.stock_data = download_stock(
            self.current_symbol, self.start_date, self.end_date
        )
        self.current_step = self.window_size
        self.current_balance = self.initial_balance
        self.shares_held = 0
        self.current_portfolio_value = self.current_balance
        self.portfolio_history = []
        self._last_action = None
        return self._next_observation(), {}

    def step(self, action: int):
        self.current_step += 1
        terminated = self.current_step >= len(self.stock_data) - 1
        truncated = False
        current_price = self.stock_data.iloc[self.current_step]["Close"]

        if action == 0 and self.current_balance >= current_price:
            self.shares_held += 1
            self.current_balance -= current_price
        elif action == 1 and self.shares_held > 0:
            self.shares_held -= 1
            self.current_balance += current_price

        self.current_portfolio_value = (
            self.current_balance + self.shares_held * current_price
        )
        self.portfolio_history.append(self.current_portfolio_value)
        reward = self.current_portfolio_value - self.initial_balance
        self._last_action = action

        return self._next_observation(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human" and self._last_action is not None:
            action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
            print(
                f"Step: {self.current_step} | "
                f"Action: {action_map.get(self._last_action, '?')} | "
                f"Balance: ${self.current_balance:,.2f} | "
                f"Shares: {self.shares_held} | "
                f"Portfolio: ${self.current_portfolio_value:,.2f}"
            )
