import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline
from typing import List, Dict, Any
import sys

# ==========================================
# --- 1. CONFIGURATION & CONSTANTS ---
# ==========================================

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

# Default config for plotting reference generation (100Hz provides smooth lines)
MOCK_CONFIG = MockConfig(freq=100.0) 

# ==========================================
# --- 2. CORE CLASSES ---
# ==========================================

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
        
        # Store start/end points
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
                p_local = np.array([[0, -GATE_WIDTH/2], [0, GATE_WIDTH/2]])
                p_rotated = (R @ p_local.T).T
                gate_x = p_rotated[:, 0] + pos[0]
                gate_y = p_rotated[:, 1] + pos[1]
                
                ax.plot(gate_x, gate_y, color=GATE_COLOR, linestyle='-', linewidth=4, alpha=0.8, zorder=5)
                ax.plot(pos[0], pos[1], marker='.', color=GATE_COLOR, markersize=8, zorder=6)
                
            elif plane == 'xz':
                w, h = GATE_WIDTH / 2, GATE_HEIGHT / 2
                corners_local = np.array([
                    [0, -w, -h], [0, w, -h], [0, w, h], [0, -w, h], [0, -w, -h] 
                ])
                
                xy_local = corners_local[:, :2]
                xy_rotated = (R @ xy_local.T).T
                
                gate_x_global = xy_rotated[:, 0] + pos[0]
                gate_z_global = corners_local[:, 2] + pos[2]
                
                ax.plot(gate_x_global, gate_z_global, color=GATE_COLOR, linestyle='-', linewidth=3, alpha=1.0, zorder=5)

# ==========================================
# --- 3. HELPER FUNCTIONS ---
# ==========================================

def _plot_single_plane(ax, run_data: dict, plotter: RacingPlotter, plane: str, 
                       norm: Normalize, cmap: str, vel_history: np.ndarray, 
                       history_runs: List[Dict[str, Any]], title: str = None):
    """
    Unified helper to plot a single plane (XY or XZ).
    Handles Reference path, Gates, History (optional), and Current Heatmap.
    """
    pos = run_data.get('pos_history')
    if pos is None or pos.ndim != 2 or pos.shape[1] < 2 or len(pos) < 2:
        return None

    # 1. Setup Data based on Plane
    if plane == 'xy':
        ref_p1, ref_p2 = plotter.ref_x, plotter.ref_y
        start_p, end_p = plotter.ref_start_xy, plotter.ref_end_xy
        label = 'Y [m]'
        points = pos[:, :2].reshape(-1, 1, 2)
        start_p_run, end_p_run = pos[0, :2], pos[-1, :2]
    else: # 'xz'
        ref_p1, ref_p2 = plotter.ref_x, plotter.ref_z
        start_p, end_p = plotter.ref_start_xz, plotter.ref_end_xz
        label = 'Z [m]'
        points = pos[:, [0, 2]].reshape(-1, 1, 2)
        start_p_run, end_p_run = pos[0, [0, 2]], pos[-1, [0, 2]]

    # 2. Plot Reference Path and Markers
    ax.plot(ref_p1, ref_p2, 'k--', linewidth=2.0, alpha=0.2, label='Reference Path')
    ax.plot(start_p[0], start_p[1], 'ro', markersize=6, zorder=12, label='Ref Start/End')
    ax.plot(end_p[0], end_p[1], 'ro', markersize=6, zorder=12)

    # 3. Plot Gates
    plotter._plot_gates_as_lines(ax, plane)
    
    # 4. Plot History Paths (Optional, typically gray)
    # Uncomment if you want history paths visible in all plots
    # for history_run in history_runs:
    #     hist_pos = history_run.get('pos_history')
    #     if hist_pos is None or len(hist_pos) < 2: continue
    #     if plane == 'xy':
    #         ax.plot(hist_pos[:, 0], hist_pos[:, 1], color='gray', alpha=0.3, linewidth=1)
    #     else:
    #         ax.plot(hist_pos[:, 0], hist_pos[:, 2], color='gray', alpha=0.3, linewidth=1)

    # 5. Plot Heatmap (Current Run)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm) 
    lc.set_array(vel_history)
    lc.set_linewidth(3)
    ax.add_collection(lc)
    
    # 6. Add Start/End Markers for TRAJECTORY
    ax.plot(start_p_run[0], start_p_run[1], 'ko', markersize=4, zorder=10, label='Run Start/End')
    ax.plot(end_p_run[0], end_p_run[1], 'ko', markersize=4, zorder=10)

    # 7. Styling
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold')
        
    ax.set_ylabel(label, fontsize=14)
    # Only set xlabel if it's the bottom plot or explicitly needed (handled by caller usually, but adding default here)
    # ax.set_xlabel('X [m]', fontsize=14) 
    
    ax.set_aspect('equal', adjustable='box') 
    ax.grid(True)

    return lc

# ==========================================
# --- 4. PLOTTING FUNCTION A: GRID COMPARISON ---
# ==========================================

def plot_controller_comparison_grid(
    experiment_map: Dict[str, Dict[int, str]], 
    plotter: RacingPlotter,
    save_file: str = "controller_comparison_grid.png"
):
    """
    Plots a grid of XY plots.
    Rows: Different Controllers (e.g., SQP, MPC)
    Columns: Different Horizon values (e.g., N=20, 25, 30)
    """
    print(f"\n--- Generating Grid Plot: {save_file} ---")
    
    controllers = list(experiment_map.keys())
    n_values = list(experiment_map[controllers[0]].keys())
    
    rows = len(controllers)
    cols = len(n_values)
    
    # 1. Load Data
    loaded_data_map = {}
    all_vels = []
    
    for ctrl in controllers:
        loaded_data_map[ctrl] = {}
        for n_val in n_values:
            filename = experiment_map[ctrl][n_val]
            try:
                data = np.load(filename, allow_pickle=True).tolist()
                loaded_data_map[ctrl][n_val] = data
                if data and len(data) > 0:
                    last_run = data[-1] 
                    vel = last_run.get('vel_history')
                    if vel is not None:
                        all_vels.append(vel)
            except Exception as e:
                print(f"  [Warn] Failed to load {filename}: {e}")
                loaded_data_map[ctrl][n_val] = []

    # Normalization
    if all_vels:
        full_vel_array = np.concatenate(all_vels)
        norm = Normalize(vmin=np.min(full_vel_array), vmax=np.max(full_vel_array))
    else:
        norm = Normalize(vmin=0, vmax=5)
    cmap = 'jet'

    # 2. Setup Plot
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10), sharex=True, sharey=True)
    
    last_lc = None
    
    for i, ctrl in enumerate(controllers):
        for j, n_val in enumerate(n_values):
            ax = axes[i, j]
            run_list = loaded_data_map[ctrl][n_val]
            
            if not run_list:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue
                
            run_to_plot = run_list[-1]
            history_runs = run_list[:-1]
            vel_history = run_to_plot.get('vel_history')

            # Use unified plotter
            last_lc = _plot_single_plane(
                ax=ax, 
                run_data=run_to_plot, 
                plotter=plotter, 
                plane='xy', 
                norm=norm, 
                cmap=cmap, 
                vel_history=vel_history, 
                history_runs=history_runs
            )
            
            # Specific Grid Labels
            if i == 0:
                ax.set_title(f"N = {n_val}", fontsize=18, fontweight='bold')
            if j == 0:
                ax.annotate(ctrl, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 15, 0),
                             xycoords=ax.yaxis.label, textcoords='offset points',
                             size=18, ha='right', va='center', rotation=90, fontweight='bold')
            if i == rows - 1:
                ax.set_xlabel("X [m]", fontsize=14)

    # 3. Layout & Legend
    plt.tight_layout(rect=[0.08, 0.08, 0.9, 0.95]) 
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    if last_lc:
        cbar = fig.colorbar(last_lc, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Velocity Magnitude [m/s]', fontsize=16)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    
    # Clean up legend items
    unique_labels['Velocity Heatmap'] = plt.Line2D([0], [0], color='darkorange', linewidth=3, label='Velocity Heatmap')
    if 'Ref Start/End' in unique_labels: del unique_labels['Ref Start/End']
    if 'Run Start/End' in unique_labels: del unique_labels['Run Start/End']
    unique_labels['Reference Start/End'] = axes[0, 0].plot([], [], 'ro', markersize=6)[0]
    unique_labels['Drone Trajectory Start/End'] = axes[0, 0].plot([], [], 'ko', markersize=4)[0]
    
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', 
               bbox_to_anchor=(0.5, 0.04), ncol=4, fontsize=14)
    
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_file)
    plt.show()
    print(f"Grid plot displayed.")


# ==========================================
# --- 5. PLOTTING FUNCTION B: SIDE-BY-SIDE ---
# ==========================================

def plot_three_runs_side_by_side(
    files_to_plot: List[str],
    titles: List[str],
    plotter: RacingPlotter,
    save_file: str = "three_runs_comparison.png"
):
    """
    Plots 3 specific files side-by-side.
    Top Row: XY Plane
    Bottom Row: XZ Plane
    """
    print(f"\n--- Generating Side-by-Side Plot: {save_file} ---")
    
    if len(files_to_plot) != 3 or len(titles) != 3:
        raise ValueError("Must provide exactly 3 file names and 3 titles.")

    # 1. Load Data
    all_data = []
    all_vels = []
    for f in files_to_plot:
        try:
            data = np.load(f, allow_pickle=True).tolist()
            if not isinstance(data, list) or not data:
                raise ValueError("Empty data")
            all_data.append(data)
            
            # Collect velocity for normalization
            last_run = data[-1]
            vel = last_run.get('vel_history')
            if vel is not None and len(vel) > 0:
                all_vels.append(vel)
        except Exception as e:
            print(f"  [Error] Could not load {f}: {e}")
            return

    if not all_vels:
        print("  [Error] No valid velocity data found.")
        return

    full_vel_array = np.concatenate(all_vels)
    norm = Normalize(vmin=np.min(full_vel_array), vmax=np.max(full_vel_array))
    cmap = 'jet' 
    
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    # 2. Setup Plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex='col') 
    
    last_lc = None 
    
    for i, (run_data_list, title) in enumerate(zip(all_data, titles)):
        run_to_color = run_data_list[-1]
        vel_history = run_to_color.get('vel_history')
        history_runs = run_data_list[:-1] 

        # --- Top Row: XY Plane ---
        ax_xy = axes[0, i]
        _plot_single_plane(
            ax=ax_xy, 
            run_data=run_to_color, 
            plotter=plotter, 
            plane='xy', 
            norm=norm, 
            cmap=cmap, 
            vel_history=vel_history, 
            history_runs=history_runs,
            title=title
        )
        # Ensure X labels show on top row
        ax_xy.tick_params(axis='x', labelbottom=True)
        ax_xy.set_xlabel("X [m]", fontsize=14)

        # --- Bottom Row: XZ Plane ---
        ax_xz = axes[1, i]
        last_lc = _plot_single_plane(
            ax=ax_xz, 
            run_data=run_to_color, 
            plotter=plotter, 
            plane='xz', 
            norm=norm, 
            cmap=cmap, 
            vel_history=vel_history, 
            history_runs=history_runs,
            title=None
        )
        ax_xz.set_xlabel("X [m]", fontsize=14)
        
        # Share Y axis for bottom row
        if i > 0:
            axes[1, i].sharey(axes[1, 0])

    # 3. Layout & Legend
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
    if last_lc:
        cbar = fig.colorbar(last_lc, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Velocity Magnitude [m/s]', fontsize=16)
    
    # Custom Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    
    if 'Run Start/End' in unique_labels: del unique_labels['Run Start/End']
    if 'Ref Start/End' in unique_labels: del unique_labels['Ref Start/End']
    
    unique_labels['Velocity Heatmap'] = axes[0, 0].plot([], [], color='darkorange', linewidth=3, label='Velocity Heatmap')[0]
    unique_labels['Drone Trajectory Start/End'] = axes[0, 0].plot([], [], 'ko', markersize=5)[0]
    unique_labels['Reference Start/End'] = axes[0, 0].plot([], [], 'ro', markersize=8)[0]
    
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', 
               bbox_to_anchor=(0.45, 0.03), ncol=5, fontsize=14)

    plt.tight_layout(rect=[0, 0.1, 0.9, 0.95]) 
    plt.savefig(save_file)
    plt.show()
    print(f"Side-by-side plot displayed.")

# ==========================================
# --- 6. MAIN EXECUTION BLOCK ---
# ==========================================

if __name__ == "__main__":
    # Initialize shared plotter
    plotter_instance = RacingPlotter(WAYPOINTS, GATE_DATA, config=MOCK_CONFIG)

    # # --- EXPERIMENT 1 CONFIGURATION: Grid Comparison ---
    # grid_experiment_files = {
    #     'SQP': {
    #         20: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/SQP_20.npy',
    #         25: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/SQP_25.npy',
    #         30: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/SQP_30.npy'
    #     },
    #     'MPC': {
    #         20: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/MPC_20.npy',
    #         25: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/MPC_25.npy',
    #         30: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/MPC_30.npy'
    #     }
    # }

    # --- EXPERIMENT 2 CONFIGURATION: Side-by-Side Comparison ---
    side_by_side_titles = ['PID', 'MPC', 'SQP-NMPC']
    side_by_side_files = [
        '/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/PID_test_run.npy', 
        '/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/sqp_nmpc_test_run.npy', 
        '/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/rti_sqp_nmpc_test_run.npy'
    ]

    # --- RUN PLOTS ---
    # Note: matplotlib plots are blocking by default. 
    # Close the first window to see the second one.
    
    # # 1. Plot Controller vs N Grid
    # plot_controller_comparison_grid(
    #     experiment_map=grid_experiment_files,
    #     plotter=plotter_instance,
    #     save_file="/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/N_comparison_grid.png"
    # )

    # 2. Plot 3-way Comparison
    plot_three_runs_side_by_side(
        files_to_plot=side_by_side_files,
        titles=side_by_side_titles,
        plotter=plotter_instance,
        save_file="/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/controller_comparison.png"
    )import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline
from typing import List, Dict, Any
import sys

# ==========================================
# --- 1. CONFIGURATION & CONSTANTS ---
# ==========================================

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

# Default config for plotting reference generation (100Hz provides smooth lines)
MOCK_CONFIG = MockConfig(freq=100.0) 

# ==========================================
# --- 2. CORE CLASSES ---
# ==========================================

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
        
        # Store start/end points
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
                p_local = np.array([[0, -GATE_WIDTH/2], [0, GATE_WIDTH/2]])
                p_rotated = (R @ p_local.T).T
                gate_x = p_rotated[:, 0] + pos[0]
                gate_y = p_rotated[:, 1] + pos[1]
                
                ax.plot(gate_x, gate_y, color=GATE_COLOR, linestyle='-', linewidth=4, alpha=0.8, zorder=5)
                ax.plot(pos[0], pos[1], marker='.', color=GATE_COLOR, markersize=8, zorder=6)
                
            elif plane == 'xz':
                w, h = GATE_WIDTH / 2, GATE_HEIGHT / 2
                corners_local = np.array([
                    [0, -w, -h], [0, w, -h], [0, w, h], [0, -w, h], [0, -w, -h] 
                ])
                
                xy_local = corners_local[:, :2]
                xy_rotated = (R @ xy_local.T).T
                
                gate_x_global = xy_rotated[:, 0] + pos[0]
                gate_z_global = corners_local[:, 2] + pos[2]
                
                ax.plot(gate_x_global, gate_z_global, color=GATE_COLOR, linestyle='-', linewidth=3, alpha=1.0, zorder=5)

# ==========================================
# --- 3. HELPER FUNCTIONS ---
# ==========================================

def _plot_single_plane(ax, run_data: dict, plotter: RacingPlotter, plane: str, 
                       norm: Normalize, cmap: str, vel_history: np.ndarray, 
                       history_runs: List[Dict[str, Any]], title: str = None):
    """
    Unified helper to plot a single plane (XY or XZ).
    Handles Reference path, Gates, History (optional), and Current Heatmap.
    """
    pos = run_data.get('pos_history')
    if pos is None or pos.ndim != 2 or pos.shape[1] < 2 or len(pos) < 2:
        return None

    # 1. Setup Data based on Plane
    if plane == 'xy':
        ref_p1, ref_p2 = plotter.ref_x, plotter.ref_y
        start_p, end_p = plotter.ref_start_xy, plotter.ref_end_xy
        label = 'Y [m]'
        points = pos[:, :2].reshape(-1, 1, 2)
        start_p_run, end_p_run = pos[0, :2], pos[-1, :2]
    else: # 'xz'
        ref_p1, ref_p2 = plotter.ref_x, plotter.ref_z
        start_p, end_p = plotter.ref_start_xz, plotter.ref_end_xz
        label = 'Z [m]'
        points = pos[:, [0, 2]].reshape(-1, 1, 2)
        start_p_run, end_p_run = pos[0, [0, 2]], pos[-1, [0, 2]]

    # 2. Plot Reference Path and Markers
    ax.plot(ref_p1, ref_p2, 'k--', linewidth=2.0, alpha=0.2, label='Reference Path')
    ax.plot(start_p[0], start_p[1], 'ro', markersize=6, zorder=12, label='Ref Start/End')
    ax.plot(end_p[0], end_p[1], 'ro', markersize=6, zorder=12)

    # 3. Plot Gates
    plotter._plot_gates_as_lines(ax, plane)
    
    # 4. Plot History Paths (Optional, typically gray)
    # Uncomment if you want history paths visible in all plots
    # for history_run in history_runs:
    #     hist_pos = history_run.get('pos_history')
    #     if hist_pos is None or len(hist_pos) < 2: continue
    #     if plane == 'xy':
    #         ax.plot(hist_pos[:, 0], hist_pos[:, 1], color='gray', alpha=0.3, linewidth=1)
    #     else:
    #         ax.plot(hist_pos[:, 0], hist_pos[:, 2], color='gray', alpha=0.3, linewidth=1)

    # 5. Plot Heatmap (Current Run)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm) 
    lc.set_array(vel_history)
    lc.set_linewidth(3)
    ax.add_collection(lc)
    
    # 6. Add Start/End Markers for TRAJECTORY
    ax.plot(start_p_run[0], start_p_run[1], 'ko', markersize=4, zorder=10, label='Run Start/End')
    ax.plot(end_p_run[0], end_p_run[1], 'ko', markersize=4, zorder=10)

    # 7. Styling
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold')
        
    ax.set_ylabel(label, fontsize=14)
    # Only set xlabel if it's the bottom plot or explicitly needed (handled by caller usually, but adding default here)
    # ax.set_xlabel('X [m]', fontsize=14) 
    
    ax.set_aspect('equal', adjustable='box') 
    ax.grid(True)

    return lc

# ==========================================
# --- 4. PLOTTING FUNCTION A: GRID COMPARISON ---
# ==========================================

def plot_controller_comparison_grid(
    experiment_map: Dict[str, Dict[int, str]], 
    plotter: RacingPlotter,
    save_file: str = "controller_comparison_grid.png"
):
    """
    Plots a grid of XY plots.
    Rows: Different Controllers (e.g., SQP, MPC)
    Columns: Different Horizon values (e.g., N=20, 25, 30)
    """
    print(f"\n--- Generating Grid Plot: {save_file} ---")
    
    controllers = list(experiment_map.keys())
    n_values = list(experiment_map[controllers[0]].keys())
    
    rows = len(controllers)
    cols = len(n_values)
    
    # 1. Load Data
    loaded_data_map = {}
    all_vels = []
    
    for ctrl in controllers:
        loaded_data_map[ctrl] = {}
        for n_val in n_values:
            filename = experiment_map[ctrl][n_val]
            try:
                data = np.load(filename, allow_pickle=True).tolist()
                loaded_data_map[ctrl][n_val] = data
                if data and len(data) > 0:
                    last_run = data[-1] 
                    vel = last_run.get('vel_history')
                    if vel is not None:
                        all_vels.append(vel)
            except Exception as e:
                print(f"  [Warn] Failed to load {filename}: {e}")
                loaded_data_map[ctrl][n_val] = []

    # Normalization
    if all_vels:
        full_vel_array = np.concatenate(all_vels)
        norm = Normalize(vmin=np.min(full_vel_array), vmax=np.max(full_vel_array))
    else:
        norm = Normalize(vmin=0, vmax=5)
    cmap = 'jet'

    # 2. Setup Plot
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10), sharex=True, sharey=True)
    
    last_lc = None
    
    for i, ctrl in enumerate(controllers):
        for j, n_val in enumerate(n_values):
            ax = axes[i, j]
            run_list = loaded_data_map[ctrl][n_val]
            
            if not run_list:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue
                
            run_to_plot = run_list[-1]
            history_runs = run_list[:-1]
            vel_history = run_to_plot.get('vel_history')

            # Use unified plotter
            last_lc = _plot_single_plane(
                ax=ax, 
                run_data=run_to_plot, 
                plotter=plotter, 
                plane='xy', 
                norm=norm, 
                cmap=cmap, 
                vel_history=vel_history, 
                history_runs=history_runs
            )
            
            # Specific Grid Labels
            if i == 0:
                ax.set_title(f"N = {n_val}", fontsize=18, fontweight='bold')
            if j == 0:
                ax.annotate(ctrl, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 15, 0),
                             xycoords=ax.yaxis.label, textcoords='offset points',
                             size=18, ha='right', va='center', rotation=90, fontweight='bold')
            if i == rows - 1:
                ax.set_xlabel("X [m]", fontsize=14)

    # 3. Layout & Legend
    plt.tight_layout(rect=[0.08, 0.08, 0.9, 0.95]) 
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    if last_lc:
        cbar = fig.colorbar(last_lc, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Velocity Magnitude [m/s]', fontsize=16)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    
    # Clean up legend items
    unique_labels['Velocity Heatmap'] = plt.Line2D([0], [0], color='darkorange', linewidth=3, label='Velocity Heatmap')
    if 'Ref Start/End' in unique_labels: del unique_labels['Ref Start/End']
    if 'Run Start/End' in unique_labels: del unique_labels['Run Start/End']
    unique_labels['Reference Start/End'] = axes[0, 0].plot([], [], 'ro', markersize=6)[0]
    unique_labels['Drone Trajectory Start/End'] = axes[0, 0].plot([], [], 'ko', markersize=4)[0]
    
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', 
               bbox_to_anchor=(0.5, 0.04), ncol=4, fontsize=14)
    
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_file)
    plt.show()
    print(f"Grid plot displayed.")


# ==========================================
# --- 5. PLOTTING FUNCTION B: SIDE-BY-SIDE ---
# ==========================================

def plot_three_runs_side_by_side(
    files_to_plot: List[str],
    titles: List[str],
    plotter: RacingPlotter,
    save_file: str = "three_runs_comparison.png"
):
    """
    Plots 3 specific files side-by-side.
    Top Row: XY Plane
    Bottom Row: XZ Plane
    """
    print(f"\n--- Generating Side-by-Side Plot: {save_file} ---")
    
    if len(files_to_plot) != 3 or len(titles) != 3:
        raise ValueError("Must provide exactly 3 file names and 3 titles.")

    # 1. Load Data
    all_data = []
    all_vels = []
    for f in files_to_plot:
        try:
            data = np.load(f, allow_pickle=True).tolist()
            if not isinstance(data, list) or not data:
                raise ValueError("Empty data")
            all_data.append(data)
            
            # Collect velocity for normalization
            last_run = data[-1]
            vel = last_run.get('vel_history')
            if vel is not None and len(vel) > 0:
                all_vels.append(vel)
        except Exception as e:
            print(f"  [Error] Could not load {f}: {e}")
            return

    if not all_vels:
        print("  [Error] No valid velocity data found.")
        return

    full_vel_array = np.concatenate(all_vels)
    norm = Normalize(vmin=np.min(full_vel_array), vmax=np.max(full_vel_array))
    cmap = 'jet' 
    
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    # 2. Setup Plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex='col') 
    
    last_lc = None 
    
    for i, (run_data_list, title) in enumerate(zip(all_data, titles)):
        run_to_color = run_data_list[-1]
        vel_history = run_to_color.get('vel_history')
        history_runs = run_data_list[:-1] 

        # --- Top Row: XY Plane ---
        ax_xy = axes[0, i]
        _plot_single_plane(
            ax=ax_xy, 
            run_data=run_to_color, 
            plotter=plotter, 
            plane='xy', 
            norm=norm, 
            cmap=cmap, 
            vel_history=vel_history, 
            history_runs=history_runs,
            title=title
        )
        # Ensure X labels show on top row
        ax_xy.tick_params(axis='x', labelbottom=True)
        ax_xy.set_xlabel("X [m]", fontsize=14)

        # --- Bottom Row: XZ Plane ---
        ax_xz = axes[1, i]
        last_lc = _plot_single_plane(
            ax=ax_xz, 
            run_data=run_to_color, 
            plotter=plotter, 
            plane='xz', 
            norm=norm, 
            cmap=cmap, 
            vel_history=vel_history, 
            history_runs=history_runs,
            title=None
        )
        ax_xz.set_xlabel("X [m]", fontsize=14)
        
        # Share Y axis for bottom row
        if i > 0:
            axes[1, i].sharey(axes[1, 0])

    # 3. Layout & Legend
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
    if last_lc:
        cbar = fig.colorbar(last_lc, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Velocity Magnitude [m/s]', fontsize=16)
    
    # Custom Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    
    if 'Run Start/End' in unique_labels: del unique_labels['Run Start/End']
    if 'Ref Start/End' in unique_labels: del unique_labels['Ref Start/End']
    
    unique_labels['Velocity Heatmap'] = axes[0, 0].plot([], [], color='darkorange', linewidth=3, label='Velocity Heatmap')[0]
    unique_labels['Drone Trajectory Start/End'] = axes[0, 0].plot([], [], 'ko', markersize=5)[0]
    unique_labels['Reference Start/End'] = axes[0, 0].plot([], [], 'ro', markersize=8)[0]
    
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', 
               bbox_to_anchor=(0.45, 0.03), ncol=5, fontsize=14)

    plt.tight_layout(rect=[0, 0.1, 0.9, 0.95]) 
    plt.savefig(save_file)
    plt.show()
    print(f"Side-by-side plot displayed.")

# ==========================================
# --- 6. MAIN EXECUTION BLOCK ---
# ==========================================

if __name__ == "__main__":
    # Initialize shared plotter
    plotter_instance = RacingPlotter(WAYPOINTS, GATE_DATA, config=MOCK_CONFIG)

    # # --- EXPERIMENT 1 CONFIGURATION: Grid Comparison ---
    # grid_experiment_files = {
    #     'SQP': {
    #         20: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/SQP_20.npy',
    #         25: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/SQP_25.npy',
    #         30: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/SQP_30.npy'
    #     },
    #     'MPC': {
    #         20: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/MPC_20.npy',
    #         25: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/MPC_25.npy',
    #         30: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/MPC_30.npy'
    #     }
    # }

    # --- EXPERIMENT 2 CONFIGURATION: Side-by-Side Comparison ---
    side_by_side_titles = ['PID', 'MPC', 'SQP-NMPC']
    side_by_side_files = [
        '/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/PID_test_run.npy', 
        '/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/sqp_nmpc_test_run.npy', 
        '/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/rti_sqp_nmpc_test_run.npy'
    ]

    # --- RUN PLOTS ---
    # Note: matplotlib plots are blocking by default. 
    # Close the first window to see the second one.
    
    # # 1. Plot Controller vs N Grid
    # plot_controller_comparison_grid(
    #     experiment_map=grid_experiment_files,
    #     plotter=plotter_instance,
    #     save_file="/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/N_comparison_grid.png"
    # )

    # 2. Plot 3-way Comparison
    plot_three_runs_side_by_side(
        files_to_plot=side_by_side_files,
        titles=side_by_side_titles,
        plotter=plotter_instance,
        save_file="/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/controller_comparison.png"
    )
