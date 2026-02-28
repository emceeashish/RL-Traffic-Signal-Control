import torch
import gymnasium as gym
import sumo_rl

print("=== 1. Checking PyTorch & GPU ===")
if torch.cuda.is_available():
    print(f"✅ Success! PyTorch is using your GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ Uh oh! PyTorch is defaulting to CPU. We might have a CUDA issue.")

print("\n=== 2. Checking SUMO-RL Environment ===")
try:
    # Notice the "sumo_rl/" added to the front of the file paths!
    env = gym.make('sumo-rl-v0', 
                   net_file='sumo_rl/nets/2way-single-intersection/single-intersection.net.xml',
                   route_file='sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                   use_gui=False, 
                   num_seconds=200)
    
    obs, info = env.reset()
    print("✅ Success! SUMO simulator loaded correctly.")
    
    # Run a few random actions to test the TraCI connection
    print("Running 5 random simulation steps...")
    for step in range(5):
        action = env.action_space.sample() # The agent picks a random traffic light phase
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step + 1} completed. Reward: {reward}")
        
        if terminated or truncated:
            break
            
    env.close()
    print("✅ Success! Agent communicated with SUMO flawlessly.")
    
except Exception as e:
    print(f"❌ Error loading SUMO-RL: {e}")