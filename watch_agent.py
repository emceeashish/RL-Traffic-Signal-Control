import time  # <--- We added this!
import gymnasium as gym
import sumo_rl
from stable_baselines3 import PPO
from sumo_rl.custom.project_observation import ProjectObservationFunction
from sumo_rl.custom.project_reward import ProjectRewardConfig, make_project_reward_fn

print("=== 1. Loading Custom Environment (With GUI ON) ===")
reward_fn = make_project_reward_fn(
    ProjectRewardConfig(
        w_wait=1.0, w_queue=0.2, w_switch=0.5, w_starve=0.01, w_emergency=10.0, 
        emergency_vtype_ids=("emergency",)
    )
)

env = gym.make('sumo-rl-v0', 
               net_file='sumo_rl/nets/2way-single-intersection/single-intersection.net.xml',
               route_file='sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
               use_gui=True, 
               num_seconds=1000, 
               observation_class=ProjectObservationFunction,
               reward_fn=reward_fn)

print("=== 2. Loading Trained Brain ===")
# device="cpu" removes that annoying yellow warning!
model = PPO.load("models/ppo_2way_emergency", env=env, device="cpu")

print("=== 3. Running Live Simulation ===")
obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # ==========================================
    # SLOW DOWN THE SIMULATION!
    # 0.05 seconds = 20 Frames Per Second (FPS)
    # Change to 0.1 to make it even slower
    # ==========================================
    time.sleep(0.05) 
    
    done = terminated or truncated

env.close()
print("Simulation finished!")
