import os
import csv
import gymnasium as gym
import sumo_rl
from stable_baselines3 import PPO
from sumo_rl.custom.project_observation import ProjectObservationFunction
from sumo_rl.custom.project_reward import ProjectRewardConfig, make_project_reward_fn

print("=== 1. Preparing Data Logger ===")
os.makedirs("results", exist_ok=True)

reward_fn = make_project_reward_fn(
    ProjectRewardConfig(
        w_wait=1.0, w_queue=0.2, w_switch=0.5, w_starve=0.01, w_emergency=10.0, 
        emergency_vtype_ids=("emergency",)
    )
)

env = gym.make('sumo-rl-v0', 
               net_file='sumo_rl/nets/2way-single-intersection/single-intersection.net.xml',
               route_file='sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
               use_gui=False, 
               num_seconds=3600, 
               observation_class=ProjectObservationFunction,
               reward_fn=reward_fn)

print("=== 2. Loading Trained AI ===")
model = PPO.load("models/ppo_2way_emergency", env=env, device="cpu")

print("=== 3. Simulating 1 Hour of Traffic... ===")
obs, info = env.reset()
done = False
total_reward = 0
step_count = 0

# 🚨 THE BULLETPROOF CSV WRITER 🚨
csv_file = open("results/ai_performance.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
# Write the Headers for the Excel columns
csv_writer.writerow(["Step", "Total Stopped Vehicles", "Total Waiting Time (s)", "Mean Waiting Time (s)"])

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Grab the hard math from SUMO and write it to the spreadsheet!
    stopped = info.get("system_total_stopped", 0)
    wait_time = info.get("system_total_waiting_time", 0)
    mean_wait = info.get("system_mean_waiting_time", 0)
    
    csv_writer.writerow([step_count, stopped, wait_time, mean_wait])
    
    total_reward += reward
    step_count += 1
    done = terminated or truncated

# Safely save and close the file
csv_file.close() 
env.close()

print("\n=== 📊 MATHEMATICAL RESULTS (1 HOUR) ===")
print(f"Total AI Score (Reward): {total_reward:.2f}")
print(f"Total Decisions Made: {step_count}")
print("✅ Detailed metrics successfully saved to 'results/ai_performance.csv'!")
