from __future__ import annotations

import uuid
from typing import Optional

from src.config.settings import Settings
from src.services.rl_service import RLService
from src.brokers.paper import PaperBroker
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BrokerService:
    """Service to orchestrate paper trading runs and provide broker-related operations."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.rl_service = RLService(settings=settings)
        # Single paper broker instance for the service lifecycle
        self.broker = PaperBroker(starting_balance=100_000.0)

    # PUBLIC_INTERFACE
    def paper_trade(
        self, policy_path: Optional[str] = None, capital: float = 100000.0, risk_per_trade: float = 0.01
    ) -> tuple[str, str]:
        """Run a basic paper trading simulation and return a run id and summary path."""
        # For this stub, we just simulate once using RL service
        results_path, _ = self.rl_service.simulate(episodes=1, policy_path=policy_path)
        run_id = str(uuid.uuid4())
        logger.info("Paper trading run %s results at %s", run_id, results_path)
        return run_id, results_path

    # PUBLIC_INTERFACE
    def get_balance(self) -> float:
        """Return current paper broker balance."""
        return self.broker.get_balance()

    # PUBLIC_INTERFACE
    def get_positions(self) -> int:
        """Return current paper broker open positions."""
        return self.broker.get_positions()

    # PUBLIC_INTERFACE
    def place_order(self, side: str, quantity: int, price: float) -> tuple[str, float, int]:
        """Place a paper order and return (order_id, new_balance, new_positions).

        Args:
            side: 'buy' or 'sell'
            quantity: number of units (must be > 0)
            price: execution price (must be > 0)

        Raises:
            ValueError: if inputs are invalid
        """
        s = side.lower().strip()
        if s not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        if price <= 0:
            raise ValueError("price must be > 0")

        order_id = self.broker.place_order(s, quantity, price)
        return order_id, self.broker.get_balance(), self.broker.get_positions()
