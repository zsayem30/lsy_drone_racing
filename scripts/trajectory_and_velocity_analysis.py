import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from typing import List, Dict, Any

# --- CONFIGURATION CONSTANTS (Keep existing) ---
WAYPOINTS = np.array([
    [1.0, 1.5, 0.6], 
    [0.8, 1.0, 0.6], 
    [0.55, -0.3, 0.6], 
    [0.0, -1.3, 1.075],
    [1.1, -0.85, 1.1], 
    [0.2, 0.5, 0.65], 
    [0.0, 1.2, 0.6], 
    [0.0, 1.2, 1.1],
    [-0.5, 0.0, 1.1]
])

GATE_WIDTH = 0.5
GATE_HEIGHT = 0.5
GATE_DATA = [
    {'pos': np.array([0.45, -0.5, 0.56]), 'yaw': 2.35},
    {'pos': np.array([1.0, -1.05, 1.11]), 'yaw': -1.28},
    {'pos': np.array([0.0, 1.0, 0.56]), 'yaw': 0.0},
    {'pos': np.array([-0.5, 0.0, 1.11]), 'yaw': 3.14},
]

class MockConfig:
    def __init__(self, freq):
        self.env = self.Env(freq)
    class Env:
        def __init__(self, freq):
            self.freq = freq
MOCK_CONFIG = MockConfig(freq=50.0) 

# --- RACING PLOTTER CLASS (Keep existing) ---

class RacingPlotter:
    """Generates visualizations based on collected trajectory data and gates."""
    def __init__(self, waypoints, gate_data: list[dict], config=MOCK_CONFIG):
        self.waypoints = waypoints
        self.gate_data = gate_data
        
        # --- CUBIC SPLINE REFERENCE PATH IMPLEMENTATION ---
        des_time = 8
        ts = np.linspace(0, des_time, waypoints.shape[0])
        ts_interp = np.linspace(0, des_time, int(config.env.freq * des_time))
        
        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])
        
        self.ref_x = cs_x(ts_interp)
        self.ref_y = cs_y(ts_interp)
        self.ref_z = cs_z(ts_interp)
        
        # Store start/end points of the interpolated reference path
        self.ref_start_xy = np.array([self.ref_x[0], self.ref_y[0]])
        self.ref_end_xy = np.array([self.ref_x[-1], self.ref_y[-1]])
        self.ref_start_xz = np.array([self.ref_x[0], self.ref_z[0]])
        self.ref_end_xz = np.array([self.ref_x[-1], self.ref_z[-1]])
            
    def _plot_gates_as_lines(self, ax, plane='xy'):
        GATE_COLOR = 'black'
        
        for gate in self.gate_data:
            pos = gate['pos']
            yaw = gate.get('yaw', 0.0)
            
            # Standard 2D Rotation Matrix
            R = np.array([[np.cos(yaw), -np.sin(yaw)],
                          [np.sin(yaw),  np.cos(yaw)]])
            
            if plane == 'xy':
                # --- XY VIEW (Top Down) ---
                p_local = np.array([[0, -GATE_WIDTH/2], [0, GATE_WIDTH/2]])
                p_rotated = (R @ p_local.T).T
                
                gate_x = p_rotated[:, 0] + pos[0]
                gate_y = p_rotated[:, 1] + pos[1]
                
                ax.plot(gate_x, gate_y, color=GATE_COLOR, linestyle='-', linewidth=4, alpha=0.8, zorder=5)
                ax.plot(pos[0], pos[1], marker='.', color=GATE_COLOR, markersize=8, zorder=6)
                
            elif plane == 'xz':
                # --- XZ VIEW (Side View) ---
                w = GATE_WIDTH / 2
                h = GATE_HEIGHT / 2
                
                corners_local = np.array([
                    [0, -w, -h],
                    [0,  w, -h],
                    [0,  w,  h],
                    [0, -w,  h],
                    [0, -w, -h] 
                ])
                
                xy_local = corners_local[:, :2]
                xy_rotated = (R @ xy_local.T).T
                
                gate_x_global = xy_rotated[:, 0] + pos[0]
                gate_z_global = corners_local[:, 2] + pos[2]
                
                ax.plot(gate_x_global, gate_z_global, 
                         color=GATE_COLOR, linestyle='-', linewidth=3, alpha=1.0, zorder=5)


def load_and_process_data(files_to_plot: List[str]):
    """Loads the last run data from each file and returns a list of data dictionaries."""
    all_runs = []
    for f in files_to_plot:
        try:
            data = np.load(f, allow_pickle=True).tolist()
            if not isinstance(data, list) or not data:
                print(f"File {f} content is invalid or empty.")
                return None
            all_runs.append(data[-1]) # Only take the last run from each file
        except FileNotFoundError:
            print(f"Error: File not found at {f}")
            return None
        except Exception as e:
            print(f"An error occurred while loading data from {f}: {e}")
            return None
    return all_runs

# --- FIGURE 1: TRAJECTORY PLOT FUNCTION ---

def plot_trajectories(all_runs: List[Dict], titles: List[str], plotter: RacingPlotter, 
                      colors: List[str], save_file: str = "/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/trajectory_comparison.png"):
    
    # Setup Figure (2 rows, 1 column: XY on top, XZ on bottom)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True, gridspec_kw={'hspace': 0.15})
    
    fig.suptitle('Controller Trajectory Comparison (Last Run)', fontsize=20, y=0.98)
    
    # --- 1. XY Plane (Top-down) - AXES[0] ---
    ax_xy = axes[0]
    
    # Plot Reference Path and Gates
    ax_xy.plot(plotter.ref_x, plotter.ref_y, 'k--', linewidth=2.0, alpha=0.3, label='Reference Path')
    plotter._plot_gates_as_lines(ax_xy, 'xy')
    
    # Plot all three trajectories
    for run_data, title, color in zip(all_runs, titles, colors):
        pos = run_data.get('pos_history')
        if pos is not None and pos.ndim == 2 and pos.shape[1] >= 2 and len(pos) >= 2:
            ax_xy.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, label=f'{title}')
            ax_xy.plot(pos[0, 0], pos[0, 1], marker='o', color=color, markersize=5, zorder=10)

    ax_xy.set_title('Top-Down View (XY Plane)', fontsize=16)
    plt.setp(ax_xy.get_xticklabels(), visible=False) # Hide X-tick labels for top plot
    ax_xy.set_ylabel('Y [m]', fontsize=14)
    ax_xy.set_aspect('equal', adjustable='box') 
    ax_xy.grid(True)
    
    
    # --- 2. XZ Plane (Side View) - AXES[1] ---
    ax_xz = axes[1]

    # Plot Reference Path and Gates
    ax_xz.plot(plotter.ref_x, plotter.ref_z, 'k--', linewidth=2.0, alpha=0.3) 
    plotter._plot_gates_as_lines(ax_xz, 'xz')

    # Plot all three trajectories
    for run_data, color in zip(all_runs, colors):
        pos = run_data.get('pos_history')
        if pos is not None and pos.ndim == 2 and pos.shape[1] >= 3 and len(pos) >= 2:
            ax_xz.plot(pos[:, 0], pos[:, 2], color=color, linewidth=2) 
            ax_xz.plot(pos[0, 0], pos[0, 2], marker='o', color=color, markersize=5, zorder=10)

    ax_xz.set_title('Side View (XZ Plane)', fontsize=16)
    ax_xz.set_xlabel('X [m]', fontsize=14) 
    ax_xz.set_ylabel('Z [m]', fontsize=14)
    ax_xz.set_aspect('equal', adjustable='box') 
    ax_xz.grid(True)
    
    
    # --- Final Layout and Legend ---
    handles, labels = ax_xy.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=14)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
    plt.savefig(save_file)
    plt.show()
    print(f"Trajectory plot saved to {save_file}")
    
# --- FIGURE 2: VELOCITY PLOT FUNCTION ---

def plot_velocity(all_runs: List[Dict], titles: List[str], config: MockConfig, 
                  colors: List[str], save_file: str = "/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/velocity_comparison.png"):
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    for run_data, title, color in zip(all_runs, titles, colors):
        vel_history = run_data.get('vel_history')
        if vel_history is not None and len(vel_history) > 1:
            time_vector = np.linspace(0, len(vel_history) / config.env.freq, len(vel_history))
            ax.plot(time_vector, vel_history, color=color, linewidth=2, label=f'{title}')

    ax.set_title('Velocity Magnitude vs. Time (Last Run)', fontsize=18)
    ax.set_xlabel('Time [s]', fontsize=16) 
    ax.set_ylabel('Velocity Magnitude [m/s]', fontsize=16)
    ax.grid(True)
    ax.legend(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_file)
    plt.show()
    print(f"Velocity plot saved to {save_file}")


# --- EXECUTION BLOCK: Load Data and Generate Outputs ---

def generate_comparison_plots(files_to_plot: List[str], titles: List[str], plotter: RacingPlotter, config: MockConfig):
    
    # Define colors globally
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e'] # blue (PID), green (MPC), orange (SQP-NMPC)

    # Load data
    all_runs = load_and_process_data(files_to_plot)
    if all_runs is None:
        return

    # Set global font size for ticks
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    # 1. Generate Trajectory Plot
    plot_trajectories(all_runs, titles, plotter, colors)
    
    # 2. Generate Velocity Plot
    plot_velocity(all_runs, titles, config, colors)
    
    # Optional: Generate a performance table if needed (kept from original context)
    # df = generate_performance_table(all_runs)
    # print("\nPerformance Metrics:")
    # print(df)


titles = ['PID', 'SQP NMPC (without RTI)', 'SQP NMPC (with RTI)']
plotter_instance = RacingPlotter(WAYPOINTS, GATE_DATA, config=MOCK_CONFIG)

all_runs_data_list = [
        '/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/PID_test_run.npy', 
        '/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/sqp_nmpc_test_run.npy', 
        '/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/rti_sqp_nmpc_test_run.npy'
    ]

# Run the function to generate and display the two separate plots
generate_comparison_plots(
    files_to_plot=all_runs_data_list,
    titles=titles,
    plotter=plotter_instance,
    config=MOCK_CONFIG
)
