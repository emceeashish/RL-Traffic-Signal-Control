#!/usr/bin/env python3
import os
from pathlib import Path

import ray
import sumo_rl

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from sumo_rl.custom.project_observation import ProjectObservationFunction
from sumo_rl.custom.project_reward import ProjectRewardConfig, make_project_reward_fn


def write_emergency_route_file(
    out_path: str,
    episode_seconds: int = 3600,
    interval_s: int = 300,
    start_time_s: int = 0,
    vtype_id: str = "emergency",
    route_edges: str = "left0A0 A0B0 B0C0 C0D0 D0right0",
):
    """Second route file: defines vType + flow injection."""
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="{vtype_id}" vClass="emergency" accel="2.6" decel="4.5" sigma="0.5"
           length="5.0" minGap="2.5" maxSpeed="25" color="1,0,0" guiShape="emergency"/>
    <route id="em_route_lr" edges="{route_edges}"/>
    <flow id="flow_emg" type="{vtype_id}" route="em_route_lr"
          begin="{start_time_s}" end="{episode_seconds}" period="{interval_s}"
          departLane="best" departSpeed="max"/>
</routes>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_content)


def main():
    # -------- Paths --------
    net_file = "sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml"
    base_route_file = "sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml"
    emergency_route_file = "generated_routes/grid4x4_emergency.rou.xml"

    episode_seconds = 3600
    interval_s = 300  

    write_emergency_route_file(
        out_path=emergency_route_file,
        episode_seconds=episode_seconds,
        interval_s=interval_s,
        start_time_s=0,
        vtype_id="emergency",
        route_edges="left0A0 A0B0 B0C0 C0D0 D0right0",
    )

    route_files_combined = f"{base_route_file},{emergency_route_file}"

    # -------- Reward --------
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

    # -------- Env config --------
    env_config = dict(
        net_file=net_file,
        route_file=route_files_combined,
        use_gui=False,
        num_seconds=episode_seconds,
        sumo_seed="random",
        observation_class=ProjectObservationFunction,
        reward_fn=reward_fn,
        additional_sumo_cmd="--no-step-log true",
    )

    def env_creator(cfg):
        pz = sumo_rl.parallel_env(**cfg)
        return ParallelPettingZooEnv(pz)

    register_env("sumo_grid4x4_pz", env_creator)

    tmp = sumo_rl.parallel_env(**env_config)
    a0 = tmp.possible_agents[0]
    obs_space = tmp.observation_space(a0)
    act_space = tmp.action_space(a0)
    tmp.close()

    ray.init(ignore_reinit_error=True)

    # -------- PPO config --------
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .environment(env="sumo_grid4x4_pz", env_config=env_config)
        .framework("torch")
        .resources(num_gpus=1)
        .training(
            lr=3e-4,
            train_batch_size=4000,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.0,
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
    )

    if hasattr(config, "env_runners"):
        config = config.env_runners(
            num_env_runners=1,
            rollout_fragment_length=20,
            sample_timeout_s=300,
            batch_mode="truncate_episodes",
        )
    else:
        config = config.rollouts(
            num_rollout_workers=1,
            rollout_fragment_length=20,
            batch_mode="truncate_episodes",
        )

    algo = config.build()

    # ----------------------------
    # 🔥 FOLDER RECOVERY LOGIC (RESUMING FROM 440k)
    # ----------------------------
    # Explicitly grab the 440k folder you accidentally stopped at
    load_dir = Path("models") / "rllib_ppo_grid4x4_shared_emergency_500k" / "checkpoint_440k"
    
    # We will keep saving in the main 500k folder so everything stays organized
    save_dir_new = Path("models") / "rllib_ppo_grid4x4_shared_emergency_500k"
    save_dir_new.mkdir(parents=True, exist_ok=True)
    
    target_timesteps = 500_000

    # ----------------------------
    # 🔥 BULLETPROOF LOAD LOGIC
    # ----------------------------
    latest_ckpt_file = None
    if (load_dir / "rllib_checkpoint.json").exists():
        latest_ckpt_file = str(load_dir.resolve())
    else:
        import glob
        candidate_files = glob.glob(str(load_dir / "checkpoint-*"))
        if candidate_files:
            latest_ckpt_file = str(Path(max(candidate_files, key=os.path.getmtime)).resolve())

    if latest_ckpt_file:
        print(f"🔄 RESCUE SUCCESSFUL! Loading existing 440k brain from: {latest_ckpt_file}")
        algo.restore(latest_ckpt_file)
    else:
        print(f"❌ FATAL ERROR: Could not find the 440k weights in {load_dir}!")
        print("Stopping script immediately to prevent starting from zero.")
        return

    print("=== Training resumed (440k -> 500k) ===")
    print(f"Loading from: {load_dir}")
    print(f"Saving to:    {save_dir_new}")

    # ----------------------------
    # 🔥 BULLETPROOF SAVE MATH
    # ----------------------------
    next_save_target = None

    while True:
        result = algo.train()
        ts_total = result.get("timesteps_total", 0)
        rew = result.get("episode_reward_mean", None)
        print(f"timesteps_total={ts_total}  episode_reward_mean={rew}")

        if next_save_target is None:
            # If it loads at 440k, the next target automatically becomes 460k
            next_save_target = ((ts_total // 20_000) + 1) * 20_000

        # Save exactly every 20k steps safely IN A NEW FOLDER
        if ts_total >= next_save_target:
            milestone_folder = save_dir_new / f"checkpoint_{next_save_target // 1000}k"
            milestone_folder.mkdir(parents=True, exist_ok=True)

            print(f"💾 Milestone reached ({ts_total} steps)! Saving to its own folder...")
            save_result = algo.save(str(milestone_folder))
            ckpt_path = getattr(save_result, "checkpoint", None)
            ckpt = ckpt_path.path if ckpt_path else save_result
            print(f"✅ Intermediate checkpoint saved safely in: {ckpt}")
            next_save_target += 20_000

        if ts_total >= target_timesteps:
            break

    # Save final in its own specific folder
    final_folder = save_dir_new / "checkpoint_500k_final"
    final_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving FINAL 500k checkpoint to {final_folder}...")
    save_result = algo.save(str(final_folder))
    ckpt_path = getattr(save_result, "checkpoint", None)
    ckpt = ckpt_path.path if ckpt_path else save_result
    print(f"=== Done. Final 500k checkpoint: {ckpt} ===")

    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()