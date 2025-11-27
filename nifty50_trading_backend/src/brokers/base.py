from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AccountState:
    balance: float
    positions: int = 0


class BrokerBase:
    """A minimal interface representing broker operations."""

    # PUBLIC_INTERFACE
    def get_balance(self) -> float:
        """Return current account balance."""
        raise NotImplementedError

    # PUBLIC_INTERFACE
    def get_positions(self) -> int:
        """Return number of open positions."""
        raise NotImplementedError

    # PUBLIC_INTERFACE
    def place_order(self, side: str, quantity: int, price: float) -> str:
        """Place an order and return an order id."""
        raise NotImplementedError
