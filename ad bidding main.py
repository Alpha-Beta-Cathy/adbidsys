import numpy as np
import pandas as pd
import plotly.express as px
from typing import Union


class AdBidQL:

    def __init__(
        self,
        initial_bid,
        initial_conversions,
        target_cpa,
        audience="prospecting",
        delta=2.0,
        value_per_conversion=45.0,
        bid_increment=0.25,
        learning_rate=0.10,
        discount_factor=0.90,
        exploration_prob=0.20,
        error_term=0.20,
        random_walk_std=1.5,
    ):

        self.initial_bid = initial_bid
        self.initial_conversions = initial_conversions
        self.target_cpa = target_cpa

        self.audience = audience

        self.delta = delta
        self.value_per_conversion = value_per_conversion

        self.bid_increment = bid_increment

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        self.error_term = error_term
        self.random_walk_std = random_walk_std

        self.num_actions = 3

        # CPA × CTR × IMPRESSION × AUDIENCE × ACTION

        self.q_values = np.zeros(
            (
                3,    # CPA
                3,    # CTR
                3,    # IMPRESSIONS
                4,    # AUDIENCE
                3,    # ACTIONS
            )
        )

        self.episode_rewards = []
        self.isfit = False

    # =========================================================================
    # STATE SPACE
    # =========================================================================

    def calculate_state(
        self,
        cpa,
        ctr,
        impressions,
        audience,
    ):

        # CPA

        if cpa < self.target_cpa * 0.85:
            cpa_state = 0
        elif cpa > self.target_cpa * 1.15:
            cpa_state = 2
        else:
            cpa_state = 1

        # CTR

        if ctr < 0.01:
            ctr_state = 0
        elif ctr < 0.03:
            ctr_state = 1
        else:
            ctr_state = 2

        # Impressions

        if impressions < 3000:
            impression_state = 0
        elif impressions < 7000:
            impression_state = 1
        else:
            impression_state = 2

        audience_map = {
            "prospecting": 0,
            "lookalike": 1,
            "broad": 2,
            "retargeting": 3,
        }

        audience_state = audience_map.get(
            audience.lower(),
            0,
        )

        return (
            cpa_state,
            ctr_state,
            impression_state,
            audience_state,
        )

    # =========================================================================
    # ENVIRONMENT
    # =========================================================================

    def calculate_ctr(
        self,
        audience,
    ):

        base_ctr = {
            "prospecting": 0.015,
            "lookalike": 0.020,
            "broad": 0.012,
            "retargeting": 0.050,
        }

        ctr = (
            base_ctr[audience]
            + np.random.normal(
                0,
                0.002,
            )
        )

        return max(
            0.001,
            ctr,
        )

    def calculate_impressions(
        self,
        bid,
    ):

        return max(
            100,
            int(
                1000
                + bid * 500
                + np.random.normal(
                    0,
                    300,
                )
            )
        )

    def calculate_clicks(
        self,
        impressions,
        ctr,
    ):

        return max(
            1,
            int(
                impressions * ctr
            )
        )

    def calculate_conversions(
        self,
        current_bid,
        current_conversions,
        new_bid,
    ):

        conversions = (
            current_conversions
            + self.delta
            * (new_bid - current_bid)
            + np.random.normal(
                0,
                self.random_walk_std,
            )
        )

        return max(
            1,
            int(
                np.floor(conversions)
            )
        )

    def calculate_cpa(
        self,
        bid,
        conversions,
    ):

        spend = bid * conversions

        return spend / conversions

    # =========================================================================
    # REWARD
    # =========================================================================

    def calculate_reward(
        self,
        bid,
        conversions,
    ):

        return (
            (self.value_per_conversion - bid)
            * conversions
            * (1 - self.error_term)
        )

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(
        self,
        num_episodes=1000,
        max_steps=100,
    ):

        self.episode_rewards = []

        for episode in range(num_episodes):

            current_bid = self.initial_bid

            current_conversions = (
                self.initial_conversions
            )

            for step in range(max_steps):

                ctr = self.calculate_ctr(
                    self.audience
                )

                impressions = (
                    self.calculate_impressions(
                        current_bid
                    )
                )

                clicks = self.calculate_clicks(
                    impressions,
                    ctr,
                )

                cpa = self.calculate_cpa(
                    current_bid,
                    current_conversions,
                )

                state = self.calculate_state(
                    cpa=cpa,
                    ctr=ctr,
                    impressions=impressions,
                    audience=self.audience,
                )

                # epsilon-greedy

                if (
                    np.random.rand()
                    < self.exploration_prob
                ):
                    action = np.random.randint(3)
                else:
                    action = np.argmax(
                        self.q_values[
                            state[0],
                            state[1],
                            state[2],
                            state[3],
                        ]
                    )

                new_bid = round(
                    current_bid
                    + (action - 1)
                    * self.bid_increment,
                    2,
                )

                if new_bid <= 0:
                    continue

                new_conversions = (
                    self.calculate_conversions(
                        current_bid,
                        current_conversions,
                        new_bid,
                    )
                )

                next_ctr = self.calculate_ctr(
                    self.audience
                )

                next_impressions = (
                    self.calculate_impressions(
                        new_bid
                    )
                )

                next_cpa = self.calculate_cpa(
                    new_bid,
                    new_conversions,
                )

                next_state = self.calculate_state(
                    cpa=next_cpa,
                    ctr=next_ctr,
                    impressions=next_impressions,
                    audience=self.audience,
                )

                reward = self.calculate_reward(
                    new_bid,
                    new_conversions,
                )

                current_q = self.q_values[
                    state[0],
                    state[1],
                    state[2],
                    state[3],
                    action,
                ]

                max_future_q = np.max(
                    self.q_values[
                        next_state[0],
                        next_state[1],
                        next_state[2],
                        next_state[3],
                    ]
                )

                self.q_values[
                    state[0],
                    state[1],
                    state[2],
                    state[3],
                    action,
                ] = (
                    current_q
                    + self.learning_rate
                    * (
                        reward
                        + self.discount_factor
                        * max_future_q
                        - current_q
                    )
                )

                current_bid = new_bid
                current_conversions = (
                    new_conversions
                )

                self.episode_rewards.append(
                    reward
                )

        self.isfit = True

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_q_table(self):
        return self.q_values

    def predict(
        self,
        input_bid,
        input_cpa,
        input_ctr,
        input_impressions,
        audience,
    ):

        state = self.calculate_state(
            cpa=input_cpa,
            ctr=input_ctr,
            impressions=input_impressions,
            audience=audience,
        )

        action = np.argmax(
            self.q_values[
                state[0],
                state[1],
                state[2],
                state[3],
            ]
        )

        return round(
            input_bid
            + (action - 1)
            * self.bid_increment,
            2,
        )

    def plot_rewards(self):

        px.line(
            self.episode_rewards,
            title="Reward Per Training Step",
            labels={
                "index": "Step",
                "value": "Reward",
            },
            template="plotly_dark",
        ).show()

