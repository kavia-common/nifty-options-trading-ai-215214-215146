from __future__ import annotations

import uuid

from src.brokers.base import BrokerBase, AccountState


class PaperBroker(BrokerBase):
    """A very simple paper broker that allows querying balance and positions."""

    def __init__(self, starting_balance: float = 100000.0):
        self.state = AccountState(balance=starting_balance, positions=0)

    # PUBLIC_INTERFACE
    def get_balance(self) -> float:
        """Return current simulated balance."""
        return self.state.balance

    # PUBLIC_INTERFACE
    def get_positions(self) -> int:
        """Return current open positions (stub)."""
        return self.state.positions

    # PUBLIC_INTERFACE
    def place_order(self, side: str, quantity: int, price: float) -> str:
        """Pretend to place an order and adjust state trivially."""
        notional = quantity * price
        if side.lower() == "buy":
            self.state.balance -= notional
            self.state.positions += quantity
        elif side.lower() == "sell":
            self.state.balance += notional
            self.state.positions -= quantity
        else:
            raise ValueError("Unsupported side; use 'buy' or 'sell'")
        return str(uuid.uuid4())
