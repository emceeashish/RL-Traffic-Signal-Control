#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    files = {
        "🛑 Baseline": "outputs/grid4x4_baseline.csv",
        "🧠 100k Brain": "outputs/grid4x4_eval_manual.csv",
        "🧠 200k Brain": "outputs/grid4x4_eval_200k.csv",
        "🧠 300k Brain": "outputs/grid4x4_eval_300k.csv",
        "🔥 400k Brain": "outputs/grid4x4_eval_400k.csv",
        "👑 500k Brain": "outputs/grid4x4_eval_500k.csv"
    }

    dfs = {}
    for name, path in files.items():
        try:
            dfs[name] = pd.read_csv(path)
        except Exception as e:
            print(f"Waiting for {name}...")

    plt.figure(figsize=(14, 8))
    
    colors = {"🛑 Baseline": "red", "🧠 100k Brain": "orange", 
              "🧠 200k Brain": "gold", "🧠 300k Brain": "dodgerblue", 
              "🔥 400k Brain": "blue", "👑 500k Brain": "lime"}
              
    widths = {"🛑 Baseline": 2, "🧠 100k Brain": 1, "🧠 200k Brain": 1, 
              "🧠 300k Brain": 1, "🔥 400k Brain": 1.5, "👑 500k Brain": 3.5}

    min_len = min([len(df) for df in dfs.values()])

    for name, df in dfs.items():
        plt.plot(df['step'][:min_len], df['system_mean_waiting_time'][:min_len], 
                 label=name, color=colors[name], linewidth=widths[name], alpha=0.85)

    plt.title('Final AI Evolution: Dumb Timers to 500,000 Steps\n(Average Waiting Time)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Simulation Step (Time)', fontsize=14)
    plt.ylabel('Average Waiting Time (Seconds)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.savefig('outputs/ultimate_evolution_500k.png', dpi=300)
    print("\n✅ Epic Master Graph saved to: outputs/ultimate_evolution_500k.png")

if __name__ == "__main__":
    main()