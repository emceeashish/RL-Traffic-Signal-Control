import os
import gymnasium as gym
import sumo_rl
from stable_baselines3 import PPO
from sumo_rl.custom.project_observation import ProjectObservationFunction
from sumo_rl.custom.project_reward import ProjectRewardConfig, make_project_reward_fn

def main():
    print("=== 1. Initializing Custom Environment ===")
    
    # Configure the custom rewards specifically for your project
    reward_fn = make_project_reward_fn(
        ProjectRewardConfig(
            w_wait=1.0, 
            w_queue=0.2, 
            w_switch=0.5, 
            w_starve=0.01,
            w_emergency=10.0, 
            emergency_vtype_ids=("emergency",)
        )
    )

    # use_gui=False makes it run blazingly fast in the background for training
    env = gym.make('sumo-rl-v0', 
                   net_file='sumo_rl/nets/2way-single-intersection/single-intersection.net.xml',
                   route_file='sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                   use_gui=False, 
                   num_seconds=100000, 
                   observation_class=ProjectObservationFunction,
                   reward_fn=reward_fn)

    print("=== 2. Building Neural Network (PPO) ===")
    # device="cuda" ensures your RTX 3090 does the heavy lifting
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device="cuda")

    print("=== 3. Starting Training Phase ===")
    # 50,000 steps is enough to prove it is learning. 
    # (For your final project submission, you might want to increase this to 200000+)
    model.learn(total_timesteps=50000)

    print("\n=== 4. Saving Learned Weights ===")
    os.makedirs("models", exist_ok=True)
    
    # This creates 'models/ppo_2way_emergency.zip'
    model.save("models/ppo_2way_emergency")
    
    print("✅ Training complete! Weights saved to 'models/ppo_2way_emergency.zip'")
    env.close()

if __name__ == "__main__":
    main()
