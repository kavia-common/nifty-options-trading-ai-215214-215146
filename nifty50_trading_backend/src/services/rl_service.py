from __future__ import annotations

import os
from typing import Optional



from src.config.settings import Settings
from src.models.datasets import load_features_csv
from src.rl.env import CrossoverPolicy, simulate_crossover
from src.storage.paths import ensure_dir, latest_file, timestamped_filename
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RLService:
    """Service orchestrating RL training (stub) and simulation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.policies_dir = ensure_dir(os.path.join(self.settings.RL_DIR, "policies"))
        self.sims_dir = ensure_dir(os.path.join(self.settings.RL_DIR, "simulations"))

    # PUBLIC_INTERFACE
    def train(self, timesteps: int = 1000, algo: str = "PPO", features_path: Optional[str] = None) -> tuple[str, str]:
        """Train a simple crossover policy; 'algo' is accepted for interface compatibility."""
        if features_path is None:
            features_path = latest_file(self.settings.FEATURES_DIR, patterns=["*.csv"])
            if features_path is None:
                raise FileNotFoundError("No features CSV found. Compute features first or provide features_path.")
        # For stub: choose default spans; could be tuned by grid-search if desired
        policy = CrossoverPolicy(short_span=12, long_span=26, threshold=0.0)
        policy_path = os.path.join(self.policies_dir, timestamped_filename("policy", "json"))
        policy.save(policy_path)
        logger.info("Saved RL policy to %s (algo=%s, timesteps=%d)", policy_path, algo, timesteps)
        return policy_path, algo

    # PUBLIC_INTERFACE
    def simulate(self, episodes: int = 1, policy_path: Optional[str] = None, features_path: Optional[str] = None) -> tuple[str, int]:
        """Simulate a policy over episodes and save the results CSV."""
        if features_path is None:
            features_path = latest_file(self.settings.FEATURES_DIR, patterns=["*.csv"])
            if features_path is None:
                raise FileNotFoundError("No features CSV found. Compute features first or provide features_path.")

        df, _ = load_features_csv(features_path)

        if policy_path is None:
            # Train a quick default policy
            policy_path, _ = self.train(timesteps=200, algo="PPO", features_path=features_path)

        policy = CrossoverPolicy.load(policy_path)
        sim_df = simulate_crossover(df, policy, episodes=episodes)

        results_path = os.path.join(self.sims_dir, timestamped_filename("simulation", "csv"))
        sim_df.to_csv(results_path, index=False)
        logger.info("Saved RL simulation results to %s", results_path)
        return results_path, episodes
