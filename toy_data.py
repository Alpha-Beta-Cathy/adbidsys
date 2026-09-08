import numpy as np
import pandas as pd
from typing import Optional


class DummyAdPerformanceGenerator:
    '''Generates a synthetic (toy) day-level ad performance dataset:
    bid, impressions, clicks, conversions, spend, ctr, cpa.

    Intended as stand-in historical data for prototyping bid-optimization
    algorithms before wiring in real ad-platform data.
    '''

    def __init__(self,
                 n_days: int = 30,
                 initial_bid: float = 10.0,
                 bid_random_walk_std: float = 0.15,
                 base_impressions: float = 5000.0,
                 impressions_bid_sensitivity: float = 200.0,
                 impressions_noise_std: float = 500.0,
                 min_impressions: float = 1000.0,
                 ctr_mean: float = 0.02,
                 ctr_std: float = 0.005,
                 ctr_min: float = 0.005,
                 ctr_max: float = 0.10,
                 conversion_rate: float = 0.15,
                 conversions_noise_std: float = 3.0,
                 min_conversions: float = 1.0,
                 random_seed: Optional[int] = None) -> None:
        '''
        Args:
            - n_days: number of days of history to simulate
            - initial_bid: starting bid the random walk begins from
            - bid_random_walk_std: day-to-day bid fluctuation
            - base_impressions: baseline impressions before the bid effect
            - impressions_bid_sensitivity: impressions gained per $1 of bid
            - impressions_noise_std: noise added to impressions
            - min_impressions: floor on impressions
            - ctr_mean, ctr_std, ctr_min, ctr_max: CTR distribution and clip range
            - conversion_rate: fraction of clicks that convert
            - conversions_noise_std: noise added to conversions
            - min_conversions: floor on conversions
            - random_seed: optional seed for reproducibility. Leave as None to
              get a different dataset every call.
        '''
        self.n_days = n_days
        self.initial_bid = initial_bid
        self.bid_random_walk_std = bid_random_walk_std
        self.base_impressions = base_impressions
        self.impressions_bid_sensitivity = impressions_bid_sensitivity
        self.impressions_noise_std = impressions_noise_std
        self.min_impressions = min_impressions
        self.ctr_mean = ctr_mean
        self.ctr_std = ctr_std
        self.ctr_min = ctr_min
        self.ctr_max = ctr_max
        self.conversion_rate = conversion_rate
        self.conversions_noise_std = conversions_noise_std
        self.min_conversions = min_conversions
        self.random_seed = random_seed

        # Populated once generate() is called
        self.df: Optional[pd.DataFrame] = None

    def generate(self) -> pd.DataFrame:
        '''Simulate n_days of ad performance and return it as a DataFrame.
        Also stores the result on self.df for get_seed_values() to use.
        '''
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        n_days = self.n_days

        dates = pd.date_range(
            end=pd.Timestamp.today().normalize(),
            periods=n_days)

        simulated_bid = (
            self.initial_bid
            + np.cumsum(np.random.normal(0, self.bid_random_walk_std, n_days)))

        simulated_impressions = np.maximum(
            self.min_impressions,
            np.round(
                self.base_impressions
                + simulated_bid * self.impressions_bid_sensitivity
                + np.random.normal(0, self.impressions_noise_std, n_days)))

        simulated_ctr = np.clip(
            np.random.normal(self.ctr_mean, self.ctr_std, n_days),
            self.ctr_min,
            self.ctr_max)

        simulated_clicks = (simulated_impressions * simulated_ctr).astype(int)

        simulated_conversions = np.maximum(
            self.min_conversions,
            np.round(
                simulated_clicks * self.conversion_rate
                + np.random.normal(0, self.conversions_noise_std, n_days)))

        simulated_spend = simulated_bid * simulated_conversions

        df = pd.DataFrame({
            "date": dates,
            "bid": simulated_bid.round(2),
            "impressions": simulated_impressions.astype(int),
            "clicks": simulated_clicks.astype(int),
            "conversions": simulated_conversions.astype(int),
            "spend": simulated_spend.round(2),
        })

        df["ctr"] = df["clicks"] / df["impressions"]
        df["cpa"] = df["spend"] / df["conversions"]

        self.df = df
        return df

    def get_seed_values(self) -> dict:
        '''Return the most recent day's values as a dict, for seeding a
        downstream agent (e.g. bid_agent = MetaBidQL(initial_bid=seed_bid, ...)).
        '''
        if self.df is None:
            raise RuntimeError("Call generate() before get_seed_values().")

        last_row = self.df.iloc[-1]
        return {
            "seed_bid": float(last_row["bid"]),
            "seed_conversions": int(last_row["conversions"]),
            "seed_impressions": int(last_row["impressions"]),
            "seed_clicks": int(last_row["clicks"]),
            "seed_ctr": float(last_row["ctr"]),
            "seed_cpa": float(last_row["cpa"]),
        }


if __name__ == "__main__":
    # Example usage
    generator = DummyAdPerformanceGenerator(n_days=90, random_seed=929)
    df = generator.generate()
    print(df.tail())

    seed_values = generator.get_seed_values()
    print(seed_values)



