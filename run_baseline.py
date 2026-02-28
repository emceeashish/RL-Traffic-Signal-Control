#!/usr/bin/env python3
import os
import traci
import sumolib
import pandas as pd
from pathlib import Path

def main():
    print("🚦 Starting Dumb Baseline Simulation (Fixed-Time Lights)...")
    
    # -------- Paths --------
    net_file = "sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml"
    base_route_file = "sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml"
    emergency_route_file = "generated_routes/grid4x4_emergency.rou.xml"
    route_files_combined = f"{base_route_file},{emergency_route_file}"

    # -------- SUMO Command --------
    # We use 'sumo' (no GUI) so it finishes in 5 seconds!
    sumo_binary = sumolib.checkBinary('sumo')
    sumo_cmd = [
        sumo_binary,
        "-n", net_file,
        "-r", route_files_combined,
        "--no-step-log", "true",
        "--waiting-time-memory", "10000"
    ]

    traci.start(sumo_cmd)
    
    metrics_data = []
    delta_time = 5  # Matches your AI script (which takes an action every 5 seconds)
    max_steps = 3600 // delta_time  # 720 total steps
    
    for step in range(1, max_steps + 1):
        # Advance the simulation by 5 seconds
        for _ in range(delta_time):
            traci.simulationStep()
            
        # Collect traffic metrics
        vehicles = traci.vehicle.getIDList()
        running = len(vehicles)
        
        if running > 0:
            waiting_times = [traci.vehicle.getWaitingTime(veh) for veh in vehicles]
            speeds = [traci.vehicle.getSpeed(veh) for veh in vehicles]
            
            total_waiting_time = sum(waiting_times)
            mean_waiting_time = total_waiting_time / running
            mean_speed = sum(speeds) / running
            stopped = sum(1 for s in speeds if s < 0.1) # Less than 0.1 m/s is "stopped"
        else:
            total_waiting_time = 0.0
            mean_waiting_time = 0.0
            mean_speed = 0.0
            stopped = 0
            
        # Save to our manual data list
        metrics_data.append({
            'step': step,
            'system_total_running': running,
            'system_total_stopped': stopped,
            'system_total_waiting_time': total_waiting_time,
            'system_mean_waiting_time': mean_waiting_time,
            'system_mean_speed': mean_speed
        })

    traci.close()
    
    # -------- Save to CSV --------
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out_path = out_dir / "grid4x4_baseline.csv"
    
    df = pd.DataFrame(metrics_data)
    df.to_csv(csv_out_path, index=False)
    print(f"✅ Baseline simulation complete! Results saved to: {csv_out_path}")

if __name__ == "__main__":
    main()