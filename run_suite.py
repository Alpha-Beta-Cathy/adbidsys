from adbidsys.toy_data import DummyAdPerformanceGenerator
from adbidsys.ad_bidding_main import AdBidQL

generator = DummyAdPerformanceGenerator(n_days=90, random_seed=929)
df = generator.generate()
seed = generator.get_seed_values()
 
print(df.tail())
 
agent = AdBidQL(
    initial_bid=seed["seed_bid"],
    initial_conversions=seed["seed_conversions"],
    target_cpa=10.0,
    audience="lookalike",
    delta=2.0,
    value_per_conversion=45.0,
)
 
agent.fit(
    num_episodes=1000,
    max_steps=100,
)
 
print("\nQ TABLE SHAPE")
print(agent.get_q_table().shape)
 
next_bid = agent.predict(
    input_bid=seed["seed_bid"],
    input_cpa=seed["seed_cpa"],
    input_ctr=seed["seed_ctr"],
    input_impressions=seed["seed_impressions"],
    audience="lookalike",
)
 
print("\nCURRENT METRICS")
print(f"Bid         : ${seed['seed_bid']:.2f}")
print(f"CPA         : ${seed['seed_cpa']:.2f}")
print(f"CTR         : {seed['seed_ctr']:.2%}")
print(f"Impressions : {seed['seed_impressions']}")
 
print(f"\nRECOMMENDED NEXT BID: ${next_bid:.2f}")
 
agent.plot_rewards()
 
