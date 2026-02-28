from __future__ import annotations
from typing import Iterable, Set
import numpy as np
from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction

def _get_sumo_connection(ts):
    sumo = getattr(ts, "sumo", None)
    if sumo is not None: return sumo
    env = getattr(ts, "env", None)
    if env is not None and getattr(env, "sumo", None) is not None: return env.sumo
    raise RuntimeError("Could not find a TraCI connection.")

def _lane_has_emergency(sumo, lane_id: str, emergency_vtype_ids: Set[str]) -> bool:
    veh_ids = sumo.lane.getLastStepVehicleIDs(lane_id)
    for vid in veh_ids:
        try:
            if sumo.vehicle.getTypeID(vid) in emergency_vtype_ids:
                return True
        except Exception:
            continue
    return False

class ProjectObservationFunction(ObservationFunction):
    WAIT_NORM_SECONDS: float = 60.0
    EMERGENCY_VTYPE_IDS: Set[str] = {"emergency"} 

    def __call__(self) -> np.ndarray:
        ts = self.ts
        phase_one_hot = [1.0 if ts.green_phase == i else 0.0 for i in range(ts.num_green_phases)]
        min_green_flag = [0.0 if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time else 1.0]
        
        density = ts.get_lanes_density()
        queue = ts.get_lanes_queue()
        lane_wait_s = ts.get_accumulated_waiting_time_per_lane()
        lane_wait_norm = [float(np.clip(w / self.WAIT_NORM_SECONDS, 0.0, 1.0)) for w in lane_wait_s]
        
        sumo = _get_sumo_connection(ts)
        emergency_flags = [
            1.0 if _lane_has_emergency(sumo, lane_id, self.EMERGENCY_VTYPE_IDS) else 0.0
            for lane_id in ts.lanes
        ]
        
        return np.array(phase_one_hot + min_green_flag + list(density) + list(queue) + lane_wait_norm + emergency_flags, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        n = self.ts.num_green_phases + 1 + 4 * len(self.ts.lanes)
        return spaces.Box(low=np.zeros(n, dtype=np.float32), high=np.ones(n, dtype=np.float32))
