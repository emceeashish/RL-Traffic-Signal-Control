from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Tuple
import numpy as np

def _get_sumo_connection(ts):
    sumo = getattr(ts, "sumo", None)
    if sumo is not None: return sumo
    env = getattr(ts, "env", None)
    if env is not None and getattr(env, "sumo", None) is not None: return env.sumo
    raise RuntimeError("Could not find TraCI connection.")

def _lane_has_emergency(sumo, lane_id: str, emergency_vtype_ids: Set[str]) -> bool:
    veh_ids = sumo.lane.getLastStepVehicleIDs(lane_id)
    for vid in veh_ids:
        try:
            if sumo.vehicle.getTypeID(vid) in emergency_vtype_ids:
                return True
        except Exception:
            continue
    return False

@dataclass
class ProjectRewardConfig:
    w_wait: float = 1.0        
    w_queue: float = 0.2       
    w_switch: float = 0.5      
    w_starve: float = 0.01     
    w_emergency: float = 10.0  # Massive penalty if emergency is blocked
    emergency_vtype_ids: Tuple[str, ...] = ("emergency",)

def make_project_reward_fn(cfg: ProjectRewardConfig = ProjectRewardConfig()):
    prev: Dict[int, Dict[str, float]] = {}
    emergency_vtype_ids = set(cfg.emergency_vtype_ids)

    def reward_fn(ts) -> float:
        key = id(ts)
        lane_wait = ts.get_accumulated_waiting_time_per_lane()
        curr_wait = float(np.sum(lane_wait))
        max_lane_wait = float(np.max(lane_wait)) if len(lane_wait) else 0.0
        curr_queue = float(ts.get_total_queued())
        curr_phase = int(ts.green_phase)
        
        sumo = _get_sumo_connection(ts)
        emergency_present = 0.0
        for lane_id in ts.lanes:
            if _lane_has_emergency(sumo, lane_id, emergency_vtype_ids):
                emergency_present = 1.0
                break
                
        if key not in prev:
            prev[key] = {"wait": curr_wait, "queue": curr_queue, "phase": float(curr_phase)}
            return 0.0
            
        reduced_wait = prev[key]["wait"] - curr_wait
        switched = 1.0 if (curr_phase != int(prev[key]["phase"])) else 0.0
        
        r = (cfg.w_wait * reduced_wait) - (cfg.w_queue * curr_queue) - (cfg.w_switch * switched) - (cfg.w_starve * max_lane_wait) + (cfg.w_emergency * emergency_present)
        
        prev[key] = {"wait": curr_wait, "queue": curr_queue, "phase": float(curr_phase)}
        return float(r)
        
    return reward_fn
