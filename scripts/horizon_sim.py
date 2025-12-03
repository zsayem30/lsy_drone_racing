from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any

import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
import numpy as np
import mujoco
import imageio


import os

# Plotting Libraries
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import splprep, splev


import pandas as pd
from scipy.interpolate import CubicSpline


# Ensure MuJoCo uses the correct backend for off-screen rendering
os.environ['MUJOCO_GL'] = 'egl' 

# Assuming lsy_drone_racing is installed/available
from lsy_drone_racing.utils import load_config, load_controller 

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

logger = logging.getLogger(__name__)

# --- CONFIGURATION CONSTANTS ---
# The waypoints provided in your prompt, defining the desired path.
WAYPOINTS =  np.array([
                [1.0, 1.5, 0.6], [0.8, 1.0, 0.6], [0.55, -0.3, 0.6], [0.0, -1.3, 1.075], [1.1, -0.85, 1.1], 
                [0.2, 0.5, 0.65], [0.0, 1.2, 0.6], [0.0, 1.2, 1.1], [-0.5, 0.0, 1.1], [-0.5, -0.5, 1.1],
            ])
GATE_WIDTH = 0.5
GATE_HEIGHT = 0.5
GATE_DATA = [
    {'pos': np.array([0.45, -0.5, 0.56]), 'yaw': 2.35},
    {'pos': np.array([1.0, -1.05, 1.11]), 'yaw': -0.78},
    {'pos': np.array([0.0, 1.0, 0.56]), 'yaw': 0.0},
    {'pos': np.array([-0.5, 0.0, 1.11]), 'yaw': 3.14},
]

class MockConfig:
    def __init__(self, freq):
        self.env = self.Env(freq)
    class Env:
        def __init__(self, freq):
            self.freq = freq
MOCK_CONFIG = MockConfig(freq=100.0) 

# --- PLOTTING CLASSES AND FUNCTIONS (Only RacingPlotter init shown) ---

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
        # ... (unchanged)
        GATE_COLOR = 'black'
        
        for gate in self.gate_data:
            pos = gate['pos']
            yaw = gate.get('yaw', 0.0)
            
            R = np.array([[np.cos(yaw), -np.sin(yaw)],
                          [np.sin(yaw),  np.cos(yaw)]])
            
            p_local = np.array([[0, -GATE_WIDTH/2], [0, GATE_WIDTH/2]])
            p_rotated = (R @ p_local.T).T
            
            gate_x = p_rotated[:, 0] + pos[0]
            gate_y = p_rotated[:, 1] + pos[1]
            gate_z = pos[2]
            
            if plane == 'xy':
                # Plot the line segment (gate opening edge)
                ax.plot(gate_x, gate_y, color=GATE_COLOR, linestyle='-', linewidth=4, alpha=0.8, zorder=5)
                # Plot the center marker
                ax.plot(pos[0], pos[1], marker='.', color=GATE_COLOR, markersize=8, zorder=6)
                
            elif plane == 'xz':
                # For XZ projection, plot vertical bar
                ax.plot([pos[0], pos[0]], 
                        [gate_z - GATE_HEIGHT/2, gate_z + GATE_HEIGHT/2], 
                        color=GATE_COLOR, linestyle='-', linewidth=4, alpha=0.8, zorder=5)
                # Plot the center marker
                ax.plot(pos[0], pos[2], marker='.', color=GATE_COLOR, markersize=8, zorder=6)


def plot_trajectory_with_velocity_heatmap_3d(
    run_data_list: list[dict], 
    plotter: RacingPlotter,
    run_index_to_color: int = -1,
    fig_title: str = "Trajectory and Velocity Heatmap (XY & XZ)",
    save_file: str = "trajectory_heatmap_3d.png"
):
    """
    Plots the trajectory with velocity heatmap in both XY (Top-down) and 
    XZ (Side) plane subplots. Subplots are arranged top-to-bottom.
    """
    
    if not run_data_list:
        return

    # --- CHANGE: Increased the height of the figure (figsize=(10, 14)) 
    # and enabled sharing the X-axis for consistent scaling.
    fig, (ax_xy, ax_xz) = plt.subplots(2, 1, figsize=(10, 14), sharex=True) 
    fig.suptitle(fig_title, fontsize=16)

    # ... (omitted setup code) ...
    idx_to_color = run_index_to_color if run_index_to_color >= 0 else len(run_data_list) + run_index_to_color
    
    # 3. Plot all trajectories
    for i, run_data in enumerate(run_data_list):
        pos = run_data.get('pos_history')
        vel = run_data.get('vel_history')

        if pos is None or pos.ndim != 2 or pos.shape[1] < 2 or len(pos) < 2:
            continue

        # Plot this run's trajectory
        if i == idx_to_color:
            # --- Velocity Heatmap Plot Setup ---
            vel_min = np.min(vel)
            vel_max = np.max(vel)
            norm = Normalize(vmin=vel_min, vmax=vel_max)
            cmap = 'jet' 
            
            # --- XY Plot (Top-down) ---
            ax = ax_xy
            
            # 1. Plot Reference Path
            ax.plot(plotter.ref_x, plotter.ref_y, 'k--', linewidth=2.0, alpha=0.2, label='Reference Path')
            
            # NEW: Add Start/End Markers for REFERENCE PATH (XY)
            ax.plot(plotter.ref_start_xy[0], plotter.ref_start_xy[1], 'ro', markersize=8, zorder=12, label='Ref Start/End')
            ax.text(plotter.ref_start_xy[0], plotter.ref_start_xy[1] + 0.08, 'Start', color='red', fontsize=12, ha='center', zorder=13)
            ax.plot(plotter.ref_end_xy[0], plotter.ref_end_xy[1], 'ro', markersize=8, zorder=12)
            ax.text(plotter.ref_end_xy[0], plotter.ref_end_xy[1] + 0.08, 'End', color='red', fontsize=12, ha='center', zorder=13)
            
            # 2. Plot Gates
            plotter._plot_gates_as_lines(ax, 'xy')
            
            # 3. Plot Heatmap (Current Run)
            points_xy = pos[:, :2].reshape(-1, 1, 2)
            segments_xy = np.concatenate([points_xy[:-1], points_xy[1:]], axis=1)
            lc_xy = LineCollection(segments_xy, cmap=cmap, norm=norm) 
            lc_xy.set_array(vel)
            lc_xy.set_linewidth(3)
            ax.add_collection(lc_xy)
            
            # 4. Add Start/End Markers for TRAJECTORY (XY) (Black circles)
            start_pos_xy = pos[0, :2]
            end_pos_xy = pos[-1, :2]
            ax.plot(start_pos_xy[0], start_pos_xy[1], 'ko', markersize=5, zorder=10, label='Run Start/End')
            ax.plot(end_pos_xy[0], end_pos_xy[1], 'ko', markersize=5, zorder=10)

            ax.set_xlabel('X [m]', fontsize=12)
            ax.set_ylabel('Y [m]', fontsize=12)
            ax.set_title("XY Plane (Top-down View)")
            
            # --- KEY CHANGE: Ensure the aspect ratio is equal for visual sizing
            ax.set_aspect('equal', adjustable='box') 
            
            ax.grid(True)
            
            # --- XZ Plot (Side View) ---
            ax = ax_xz
            
            # 1. Plot Reference Path
            ax.plot(plotter.ref_x, plotter.ref_z, 'k--', linewidth=2.0, alpha=0.2, label='Reference Path')
            
            # NEW: Add Start/End Markers for REFERENCE PATH (XZ)
            ax.plot(plotter.ref_start_xz[0], plotter.ref_start_xz[1], 'ro', markersize=8, zorder=12, label='Ref Start/End')
            ax.text(plotter.ref_start_xz[0], plotter.ref_start_xz[1] + 0.08, 'Start', color='red', fontsize=12, ha='center', zorder=13)
            ax.plot(plotter.ref_end_xz[0], plotter.ref_end_xz[1], 'ro', markersize=8, zorder=12)
            ax.text(plotter.ref_end_xz[0], plotter.ref_end_xz[1] + 0.08, 'End', color='red', fontsize=12, ha='center', zorder=13)
            
            # 2. Plot Gates
            plotter._plot_gates_as_lines(ax, 'xz')
            
            # 3. Plot Heatmap (X and Z coordinates)
            points_xz = pos[:, [0, 2]].reshape(-1, 1, 2)
            segments_xz = np.concatenate([points_xz[:-1], points_xz[1:]], axis=1)
            lc_xz = LineCollection(segments_xz, cmap=cmap, norm=norm) 
            lc_xz.set_array(vel)
            lc_xz.set_linewidth(3)
            line = ax.add_collection(lc_xz) # Store this for the colorbar
            
            # 4. Add Start/End Markers for TRAJECTORY (XZ) (Black circles)
            start_pos_xz = pos[0, [0, 2]]
            end_pos_xz = pos[-1, [0, 2]]
            ax.plot(start_pos_xz[0], start_pos_xz[1], 'ko', markersize=5, zorder=10, label='Run Start/End')
            ax.plot(end_pos_xz[0], end_pos_xz[1], 'ko', markersize=5, zorder=10)
            
            ax.set_xlabel('X [m]', fontsize=12)
            ax.set_ylabel('Z [m]', fontsize=12)
            ax.set_title("XZ Plane (Side View)")
            
            # --- KEY CHANGE: Ensure the aspect ratio is equal for visual sizing
            ax.set_aspect('equal', adjustable='box') 
            
            ax.grid(True)
            
            # Add a single colorbar below the two subplots
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02]) # Adjusted height and position
            cbar = fig.colorbar(line, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Velocity Magnitude [m/s]', fontsize=14)
            
        else:
            # --- Gray History Plot ---
            # Plot historical paths in both subplots
            ax_xy.plot(pos[:, 0], pos[:, 1], color='gray', alpha=0.3, linewidth=1, label='History Path')
            ax_xz.plot(pos[:, 0], pos[:, 2], color='gray', alpha=0.3, linewidth=1, label='History Path')

    # Final Legend for the entire figure (unchanged cleanup)
    handles, labels = ax_xy.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    
    if 'Run Start/End' in unique_labels:
        del unique_labels['Run Start/End']
    if 'Ref Start/End' in unique_labels:
        del unique_labels['Ref Start/End']
    
    unique_labels['Velocity Heatmap'] = ax_xy.plot([], [], color='darkorange', linewidth=3, label='Velocity Heatmap')[0]
    unique_labels['Run Trajectory Start/End'] = ax_xy.plot([], [], 'ko', markersize=5)[0]
    unique_labels['Reference Start/End'] = ax_xy.plot([], [], 'ro', markersize=8)[0]


    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=4)

    plt.tight_layout(rect=[0, 0.13, 1, 0.95]) # Adjusted rect for bottom elements
    plt.savefig(save_file)
    plt.show()
    
    print(f"Plot saved to {save_file}")
    
    
# ... (generate_performance_table function omitted for brevity) ...
def generate_performance_table(all_runs_data: List[Dict[str, Any]]) -> pd.DataFrame:
    # ... (function body unchanged) ...
    table_data = []
    for i, run_data in enumerate(all_runs_data):
        lap_data = {
            "Lap": i + 1,
            "Lap Time [s]": f"{run_data.get('lap_time', np.nan):.2f}",
            "Success": run_data.get('success', False),
            "Gates Passed": int(run_data.get('gates_passed', 0)),
            "RMSE Tracking Error": f"{run_data.get('rmse_tracking_error', np.nan):.4f}",
            "Avg Solver Time [ms]": f"{run_data.get('avg_solver_time_ms', np.nan):.2f}",
        }
        table_data.append(lap_data)

    df = pd.DataFrame(table_data)
    df = df.set_index("Lap")
    
    return df

# --- EXECUTION BLOCK: Load Data and Generate Outputs ---

# --- METRIC AND PLOTTING CLASSES ---

class MetricTracker:
    """Collects and summarizes performance data for one episode."""
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.reset()

    def reset(self):
        self.positions = []
        self.velocities = []
        self.solver_times = []
        self.ref_errors = []
        self.success = False
        self.gates_passed = 0
        self.lap_time = 0.0

    def update(self, pos, vel, solver_dt):
        """Records data for a single time step."""
        self.positions.append(pos)
        self.velocities.append(np.linalg.norm(vel)) # Magnitude of velocity
        self.solver_times.append(solver_dt * 1000) # Convert to ms
        
        # Tracking Error Proxy: Distance to the nearest waypoint point
        dist_to_path = np.min(np.linalg.norm(self.waypoints - pos, axis=1))
        self.ref_errors.append(dist_to_path)

    def get_summary(self):
        """Returns aggregated metrics for the episode."""
        return {
            "pos_history": np.array(self.positions),
            "vel_history": np.array(self.velocities),
            "avg_solver_time_ms": np.mean(self.solver_times) if self.solver_times else 0,
            "max_solver_time_ms": np.max(self.solver_times) if self.solver_times else 0,
            # RMSE of the tracking error
            "rmse_tracking_error": np.sqrt(np.mean(np.array(self.ref_errors)**2)) if self.ref_errors else 0,
            "lap_time": self.lap_time,
            "success": self.success,
            "gates_passed": self.gates_passed
        }



# --- MAIN SIMULATION FUNCTION ---

def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 3,
    N: int | None = None,
    gui: bool | None = None,
    record_video: bool = False,
    save_summary_only: bool = False,
    save_file_as: str = "default_run",
    disturbance_scale: float = 0.0,
    plot_mode: str = "evolution" # 'comparison' or 'evolution'
) -> List[Dict]:
    """Evaluate the drone controller over multiple episodes and log metrics."""
    
    config_path = Path(__file__).parents[1] / "config" / config
    config_obj = load_config(config_path)
    

    # ... [GUI setup and Controller/Env loading remain similar] ...
    if gui is None:
        gui = config_obj.sim.gui
    else:
        config_obj.sim.gui = gui

    # Optionally override the controller prediction horizon N from the CLI.
    # This is picked up in controllers like AttitudeMPC / AttitudeMPCRTI via config.controller.N.
    if N is not None:
        try:
            N_int = int(N)
            ctrl_cfg = getattr(config_obj, "controller", None)
            # Handle both single-controller config ([controller]) and potential multi-controller ([[controller]])
            if isinstance(ctrl_cfg, list):
                for idx, c in enumerate(ctrl_cfg):
                    setattr(c, "N", N_int)
                print(f"[sim] Set controller[*].N = {N_int} from CLI")
            elif ctrl_cfg is not None:
                setattr(ctrl_cfg, "N", N_int)
                # Try to show which controller this applies to
                ctrl_file = getattr(ctrl_cfg, "file", controller)
                print(f"[sim] Set controller.N = {N_int} from CLI for '{ctrl_file}'")
            else:
                print(f"[sim] WARNING: Config has no 'controller' section; cannot apply N={N_int}")
        except Exception as e:
            print(f"[sim] WARNING: Failed to set prediction horizon N from CLI: {e}")
            
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config_obj.controller.file)
    controller_cls = load_controller(controller_path)

    save_path = Path(__file__).parents[1] / "lsy_drone_racing/experiments"
    save_path.mkdir(parents=True, exist_ok=True)

    env: DroneRaceEnv = gymnasium.make(
        config_obj.env.id,
        freq=config_obj.env.freq,
        sim_config=config_obj.sim,
        sensor_range=config_obj.env.sensor_range,
        control_mode=config_obj.env.control_mode,
        track=config_obj.env.track,
        disturbances=config_obj.env.get("disturbances"),
        randomizations=config_obj.env.get("randomizations"),
        seed=config_obj.env.seed,
    )
    env = JaxToNumpy(env)
    
    # --- VIDEO RECORDING SETUP ---
    renderer = None
    mj_data_cpu = None
    frames = []
    recording_fps = 30
    # Calculate how many simulation steps to skip to achieve ~30 FPS video
    # e.g., if sim freq is 100Hz and video is 30fps, skip ~3 steps per frame
    record_interval = max(1, int(config_obj.env.freq / recording_fps))

    if record_video:
        print(f"Initializing Renderer for recording as {save_path}/{save_file_as}.mp4")

        # Access the underlying MuJoCo model from the unwrapped env
        mj_model = env.unwrapped.sim.mj_model
        mj_data_cpu = mujoco.MjData(mj_model) # Create a CPU data structure
        renderer = mujoco.Renderer(mj_model, height=480, width=640)
    # -----------------------------


    tracker = MetricTracker(WAYPOINTS)
    ep_results = []
    
    all_runs_pos = []
    all_runs_vel = []

    for run_idx in range(n_runs):
        tracker.reset()
        obs, info = env.reset()
        # Re-instantiate controller for each run if it's IL MPC, 
        # or handle learning updates via its internal logic
        controller_obj: Controller = controller_cls(obs, info, config_obj) 
        i = 0
        sim_fps = 60 # Used for GUI throttling

        print(f"--- Starting Run {run_idx + 1}/{n_runs} for {controller or 'Default'} ---")

        while True:
            curr_time = i / config_obj.env.freq

            # --- MEASURE SOLVER TIME ---
            t_start = time.perf_counter()
            action = controller_obj.compute_control(obs, info)
            t_end = time.perf_counter()
            solver_dt = t_end - t_start
            
            # Update Metrics
            tracker.update(obs["pos"], obs["vel"], solver_dt) # Assuming single drone [0]
            
            # Step Env
            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller_obj.step_callback(action, obs, reward, terminated, truncated, info)

            # --- CAPTURE FRAME ---
            if record_video and (i % record_interval == 0):
                # 1. Get JAX/GPU state
                raw_sim_data = env.unwrapped.sim.data
                # Assuming single environment (index 0)
                jax_pos = raw_sim_data.states.pos[0]
                jax_quat = raw_sim_data.states.quat[0]
                jax_mocap_pos = raw_sim_data.mjx_data.mocap_pos
                jax_mocap_quat = raw_sim_data.mjx_data.mocap_quat

                # 2. Sync to CPU MuJoCo Data
                # Sync Drone (Assumes drone is the first set of joints)
                mj_data_cpu.qpos[:3] = np.array(jax_pos)
                mj_data_cpu.qpos[3:7] = np.array(jax_quat)
                
                # Sync Gates/Obstacles (Mocap bodies)
                # Mocap data usually covers all envs in MJX, we just need the slice for the active objects
                # However, usually the layout is consistent. We simply copy the arrays.
                if jax_mocap_pos is not None:
                    mj_data_cpu.mocap_pos[:] = np.array(jax_mocap_pos)
                    mj_data_cpu.mocap_quat[:] = np.array(jax_mocap_quat)

                # 3. Update Geometry and Render
                mujoco.mj_forward(env.unwrapped.sim.mj_model, mj_data_cpu)
                renderer.update_scene(mj_data_cpu)
                frames.append(renderer.render())
            # ---------------------
            
            if terminated or truncated or controller_finished:
                # Log success logic
                gates_passed = obs["target_gate"]
                if gates_passed == -1: 
                    gates_passed = len(config_obj.env.track.gates)
                    tracker.success = True
                else:
                    tracker.success = False
                
                tracker.gates_passed = gates_passed
                tracker.lap_time = curr_time
                break

            if config_obj.sim.gui:
                if ((i * sim_fps) % config_obj.env.freq) < sim_fps:
                    env.render()
            i += 1

        controller_obj.episode_callback()
        controller_obj.episode_reset()
        
        # Save metrics and trajectory history for this run
        summary = tracker.get_summary()
        log_episode_stats(obs, info, config_obj, curr_time)
        ep_results.append(summary)
        all_runs_pos.append(summary["pos_history"])
        all_runs_vel.append(summary["vel_history"])

        # Print Quick Stats
        print(f"Run {run_idx+1}: Time={summary['lap_time']:.2f}s | "
              f"Success={summary['success']} | "
              f"RMSE={summary['rmse_tracking_error']:.3f} | "
              f"AvgSolver={summary['avg_solver_time_ms']:.2f}ms")

    # Save video after all runs (or move inside loop to save per run)
    if record_video and len(frames) > 0:
        print(f"Saving video with {len(frames)} frames to {save_path}...")
        imageio.mimsave(f"{save_path}/{save_file_as}.mp4", frames, fps=recording_fps)
        print("Video saved!")


    # --- FINAL REPORT AND PLOTTING ---
    print("\n" + "="*40)
    print(f"FINAL AGGREGATE RESULTS FOR: {controller or 'Default'}")
    print("="*40)
    
    # Calculate aggregate metrics
    success_runs = [r for r in ep_results if r['success']]
    
    avg_lap = np.mean([r['lap_time'] for r in success_runs]) if success_runs else np.nan
    success_rate = np.mean([1 if r['success'] else 0 for r in ep_results]) * 100
    avg_rmse = np.mean([r['rmse_tracking_error'] for r in ep_results])
    avg_solver = np.mean([r['avg_solver_time_ms'] for r in ep_results])
    
    print(f"Success Rate:         {success_rate:.1f}%")
    print(f"Avg Lap Time (Valid): {avg_lap:.4f} s")
    print(f"Avg Tracking RMSE:    {avg_rmse:.4f} m")
    print(f"Avg Solver Time:      {avg_solver:.4f} ms")
    print("="*40 + "\n")

    # print(ep_results)
    
    final_summary = [avg_lap, success_rate, avg_rmse, avg_solver]
    
    if save_summary_only:
      print("summary stats stored only in the format: success rate, avg lap time, avg tracking RMSE, avg solver time: ")
      np.save(f"{save_path}/experiment_horizon/{save_file_as}/{save_file_as}_horizon_{N}_summary.npy", np.array(final_summary))
    else:
      np.save(f"{save_path}/{save_file_as}.npy", np.array(ep_results))


      all_runs_data = np.load(f"{save_path}/{save_file_as}.npy", allow_pickle=True).tolist()
      if all_runs_data:
          performance_df = generate_performance_table(all_runs_data)
          
          print("\n## Performance Metrics Per Lap ")
          print("-" * 70)
          print(performance_df.to_markdown(numalign="left", stralign="left"))
          print("-" * 70)
      else:
          print("No data available to generate the table or plots.")
          
      # 3. Generate the 3D velocity heatmap plot (XY and XZ planes)
      if all_runs_data:
          plotter_example = RacingPlotter(WAYPOINTS, GATE_DATA, config=MOCK_CONFIG)
          plot_trajectory_with_velocity_heatmap_3d(
              all_runs_data, 
              plotter_example, 
              run_index_to_color=-1, # Plot the last run as the heatmap
              fig_title="Final Lap Velocity Heatmap (XY and XZ Planes)",
              save_file=f"{save_path}/{save_file_as}_plot.png"
          )


    env.close()
    return ep_results

def log_episode_stats(obs: dict, info: dict, config_obj: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:
        gates_passed = len(config_obj.env.track.gates)
    finished = gates_passed == len(config_obj.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    
    # Example usage:
    # 1. To run IL MPC for 10 laps and plot the evolution (Like image_ca1ae8.png):
    # python analysis_script.py simulate --controller="il_mpc_controller.py" --n_runs=10 
    
    # 2. To run Linear MPC (1 lap) and plot XY/XZ path comparison (Like image_ca6ca3.png):
    # python analysis_script.py simulate --controller="linear_mpc_controller.py" --n_runs=1
    
    # 3. To test robustness of SQP MPC:
    # python analysis_script.py simulate --controller="sqp_mpc_controller.py" --n_runs=5 --save_file_as=""
    
    fire.Fire(simulate, serialize=lambda _: None)


