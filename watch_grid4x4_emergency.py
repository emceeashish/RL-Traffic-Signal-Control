#!/usr/bin/env python3
import os
import glob
from pathlib import Path

import ray
import sumo_rl
import pandas as pd  # 🔥 Added pandas to force-save the CSV
from ray.rllib.algorithms.ppo import PPOConfig

from sumo_rl.custom.project_observation import ProjectObservationFunction
from sumo_rl.custom.project_reward import ProjectRewardConfig, make_project_reward_fn

def main():
    # -------- Paths --------
    net_file = "sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml"
    base_route_file = "sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml"
    emergency_route_file = "generated_routes/grid4x4_emergency.rou.xml"
    route_files_combined = f"{base_route_file},{emergency_route_file}"

    # Ensure output directory for results exists
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔥 Changed to 500k CSV filename!
    csv_out_path = str(out_dir.resolve() / "grid4x4_eval_500k.csv") 

    # -------- Reward (Must match training) --------
    reward_fn = make_project_reward_fn(
        ProjectRewardConfig(
            w_wait=1.0,
            w_queue=0.2,
            w_switch=0.5,
            w_starve=0.01,
            w_emergency=10.0,
            emergency_vtype_ids=("emergency",),
        )
    )

    # -------- Env Config --------
    env_config = dict(
        net_file=net_file,
        route_file=route_files_combined,
        use_gui=True,  
        num_seconds=3600,
        sumo_seed="random",
        observation_class=ProjectObservationFunction,
        reward_fn=reward_fn,
    )

    ray.init(ignore_reinit_error=True)
    env = sumo_rl.parallel_env(**env_config)
    
    a0 = env.possible_agents[0]
    obs_space = env.observation_space(a0)
    act_space = env.action_space(a0)

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .framework("torch")
    )
    
    if hasattr(config, "env_runners"):
        config = config.env_runners(num_env_runners=0)
    else:
        config = config.rollouts(num_rollout_workers=0)

    config = config.multi_agent(
        policies={"shared_policy": (None, obs_space, act_space, {})},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
    )
    
    algo = config.build()

    # -------- Find and Load the Checkpoint --------
    # 🔥 Changed to load specifically from the 500k final folder!
    save_dir = Path("models") / "rllib_ppo_grid4x4_shared_emergency_500k" / "checkpoint_500k_final"
    
    if (save_dir / "rllib_checkpoint.json").exists():
        latest_ckpt_file = str(save_dir.resolve())
    else:
        candidate_files = glob.glob(str(save_dir / "checkpoint-*"))
        if candidate_files:
            latest_ckpt_file = str(Path(max(candidate_files, key=os.path.getmtime)).resolve())
        else:
            print(f"❌ ERROR: No saved weights found in {save_dir}!")
            return

    print(f"🔄 Loading brain from: {latest_ckpt_file}")
    algo.restore(latest_ckpt_file)

    # -------- The Visualization Loop --------
    print("🚦 Starting Simulation! Check the SUMO GUI window.")
    
    observations, infos = env.reset()
    
    metrics_data = []  # 🔥 List to manually store all traffic data
    step_counter = 0

    try:
        while env.agents:
            actions = {}
            for agent_id, agent_obs in observations.items():
                action = algo.compute_single_action(
                    observation=agent_obs,
                    policy_id="shared_policy",
                    explore=False
                )
                actions[agent_id] = action
                
            observations, rewards, terminations, truncations, infos = env.step(actions)
            step_counter += 1

            # 🔥 Extract the data directly from the simulation at every step
            if infos and len(infos) > 0:
                first_agent = list(infos.keys())[0]
                step_info = infos[first_agent].copy()
                step_info['step'] = step_counter
                metrics_data.append(step_info)

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped early by user!")
        
    finally:
        # 🔥 Force save the data to CSV no matter what happens
        print("💾 Saving CSV data...")
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            df.to_csv(csv_out_path, index=False)
            print(f"✅ SUCCESS! Results literally forced to save at: {csv_out_path}")
        else:
            print("⚠️ No data was collected (did it crash immediately?)")
            
        env.close()
        ray.shutdown()

if __name__ == "__main__":
    main()