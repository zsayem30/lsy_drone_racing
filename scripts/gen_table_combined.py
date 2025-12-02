import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union

# ==========================================
# --- 1. UNIFIED PERFORMANCE TABLE GENERATION ---
# ==========================================

def generate_performance_table(
    all_runs_data: List[Dict[str, Any]], 
    solver_name: str, 
    horizon_n: Union[int, None] = None
) -> pd.DataFrame:
    """
    Extracts key performance metrics from a list of lap results, converts solver time
    from milliseconds to seconds, and creates a DataFrame suitable for multi-index comparison.

    Args:
        all_runs_data: A list of dicts containing performance metrics for one lap/run.
        solver_name: A string identifying the controller type (e.g., 'SQP').
        horizon_n: The prediction horizon (N) used. If None (for PID), it's excluded from the index.

    Returns:
        A Pandas DataFrame summarizing the performance.
    """
    
    table_data = []
    for i, run_data in enumerate(all_runs_data):
        # Retrieve avg solver time in milliseconds, convert to seconds
        solver_time_ms = run_data.get('avg_solver_time_ms', np.nan)
        # Use division only if the value is not already nan, to prevent RuntimeWarning
        solver_time_s = solver_time_ms / 1000.0 if not pd.isna(solver_time_ms) else np.nan
        
        lap_data = {
            "Solver": solver_name,
            # Lap time in seconds (kept as float)
            "Lap Time [s]": run_data.get('lap_time', np.nan),
            # Success status (e.g., True/False)
            "Success": run_data.get('success', False),
            # Number of gates passed
            "Gates Passed": run_data.get('gates_passed', 0),
            # RMSE of the tracking error (kept as float)
            "RMSE Tracking Error": run_data.get('rmse_tracking_error', np.nan),
            # Converted Solver Time (seconds)
            "Solver Time [s]": solver_time_s,
            "Lap": i + 1, # Lap number
        }
        
        if horizon_n is not None:
            lap_data["Horizon N"] = horizon_n
            
        table_data.append(lap_data)

    df = pd.DataFrame(table_data)
    
    # Set the MultiIndex based on whether Horizon N is present
    if horizon_n is not None:
        df = df.set_index(["Solver", "Horizon N", "Lap"])
    else:
        df = df.set_index(["Solver", "Lap"])
    
    return df

# ==========================================
# --- 2. PLOTTING FUNCTION A: PID/MPC/SQP Comparison (Script 1) ---
# ==========================================

def plot_controller_comparison_pid_mpc_sqp(files_to_load: Dict[str, str]):
    print("## 🏆 Controller Type Comparison (PID vs MPC vs SQP) 🏆")
    print("-" * 60)
    
    all_performance_dfs = []
    
    # Load and process data
    for solver_name, file_path in files_to_load.items():
        try:
            run_data = np.load(file_path, allow_pickle=True).tolist()
            if run_data:
                df = generate_performance_table(run_data, solver_name)
                all_performance_dfs.append(df)
                print(f"- Successfully processed **{len(run_data)}** runs for **{solver_name}**.")
            else:
                print(f"- Data for **{solver_name}** is empty or corrupted and was skipped.")
        except FileNotFoundError:
            print(f"- **Error:** File not found at path: `{file_path}`. Skipping this solver.")
        except Exception as e:
            print(f"- **Error:** An unexpected error occurred loading data for **{solver_name}**: {e}")
            
    print("-" * 60)
    
    if not all_performance_dfs:
        print("No data loaded for comparison.")
        return

    raw_comparison_df = pd.concat(all_performance_dfs)

    ## --- Full Lap-by-Lap Table ---
    comparison_df = raw_comparison_df.copy()
    comparison_df["Lap Time [s]"] = comparison_df["Lap Time [s]"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else 'NaN')
    comparison_df["RMSE Tracking Error"] = comparison_df["RMSE Tracking Error"].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else 'NaN')
    comparison_df["Solver Time [s]"] = comparison_df["Solver Time [s]"].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else 'N/A')
    
    print("\n### 📋 Full Lap-by-Lap Performance Table 📋")
    print(comparison_df.to_markdown(numalign="left", stralign="left"))
    print("-" * 60)
    
    ## --- Quick Summary Table ---
    print("\n### 📊 Quick Summary of Successful Runs (Averages) 📊")
    
    summary_df = raw_comparison_df[raw_comparison_df['Success'] == True].groupby('Solver').agg({
        "Lap Time [s]": ['mean', 'min', 'count'],
        "RMSE Tracking Error": 'mean',
        "Solver Time [s]": 'mean'
    })
    
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    
    summary_df = summary_df.rename(columns={
        'Lap Time [s]_count': 'Successful Runs',
        'Lap Time [s]_mean': 'Avg Lap Time [s]',
        'Lap Time [s]_min': 'Best Lap Time [s]',
        'RMSE Tracking Error_mean': 'Avg RMSE Error',
        'Solver Time [s]_mean': 'Avg Solver Time [s]'
    })

    for col in summary_df.columns:
        if 'Time' in col or 'RMSE' in col:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.float64)) else x)
    
    print(summary_df.to_markdown(numalign="left", stralign="left"))
    print("-" * 60)

# ==========================================
# --- 3. PLOTTING FUNCTION B: N-Horizon Comparison (Script 2) ---
# ==========================================

def plot_horizon_comparison_sqp_mpc(experiment_files: Dict[str, Dict[int, str]]):
    print("## 🏆 Performance Comparison by Prediction Horizon (N) 🏆")
    print("### Metrics (Rows) vs. (Solver, Horizon) (Columns)")
    print("-" * 100)
    
    all_performance_dfs = []
    
    # Load and process data
    for solver_name, horizon_data in experiment_files.items():
        for horizon_n, file_path in horizon_data.items():
            try:
                run_data = np.load(file_path, allow_pickle=True).tolist()
                if run_data:
                    # Pass the horizon_n argument to generate_performance_table
                    df = generate_performance_table(run_data, solver_name, horizon_n)
                    all_performance_dfs.append(df)
                    print(f"- Processed **{len(run_data)}** runs for **{solver_name} (N={horizon_n})**.")
                else:
                    print(f"- Data for **{solver_name} (N={horizon_n})** is empty and skipped.")
            except FileNotFoundError:
                print(f"- **Error:** File not found at `{file_path}`. Skipping.")
            except Exception as e:
                print(f"- **Error:** Unexpected error loading data for {solver_name} (N={horizon_n}): {e}")
                
    print("-" * 100)
    
    if not all_performance_dfs:
        print("No data loaded for comparison.")
        return

    raw_comparison_df = pd.concat(all_performance_dfs)

    # Create the Aggregated Summary DataFrame (Successful Runs)
    summary_df = raw_comparison_df[raw_comparison_df['Success'] == True].groupby(['Solver', 'Horizon N']).agg({
        "Lap Time [s]": ['mean', 'min', 'count'],
        "RMSE Tracking Error": 'mean',
        "Solver Time [s]": 'mean'
    })
    
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    
    summary_df = summary_df.rename(columns={
        'Lap Time [s]_count': 'Successful Runs',
        'Lap Time [s]_mean': 'Avg Lap Time [s]',
        'Lap Time [s]_min': 'Best Lap Time [s]',
        'RMSE Tracking Error_mean': 'Avg RMSE Tracking Error',
        'Solver Time [s]_mean': 'Avg Solver Time [s]'
    })

    # Transpose the DataFrame (Metrics become rows, experiments become columns)
    transposed_summary_df = summary_df.T
    
    # Format the numerical values in the transposed table
    for idx in transposed_summary_df.index:
        for col in transposed_summary_df.columns:
            value = transposed_summary_df.loc[idx, col]
            if pd.isna(value):
                transposed_summary_df.loc[idx, col] = 'NaN'
            elif 'Time' in idx or 'RMSE' in idx:
                transposed_summary_df.loc[idx, col] = f"{value:.4f}"
            elif 'Successful Runs' in idx:
                transposed_summary_df.loc[idx, col] = f"{int(round(value))}"

    # Print the resulting comparison table
    print(transposed_summary_df.to_markdown(numalign="left", stralign="left"))
    
    print("-" * 100)


# ==========================================
# --- 4. EXECUTION BLOCK ---
# ==========================================

if __name__ == "__main__":
    # --- CONFIGURATION 1: PID, MPC, SQP Comparison ---
    pid_mpc_sqp_files = {
        "SQP NMPC": "scripts/pid_vs_mpc_sqp/10lap_sqp_nmpc.npy",
        "PID Controller": "scripts/pid_vs_mpc_sqp/10lap_pid.npy",
        "MPC": "scripts/pid_vs_mpc_sqp/10lap_mpc.npy",
    }
    
    # --- CONFIGURATION 2: SQP vs. MPC vs. Horizon N Comparison ---
    horizon_experiment_files = {
        'SQP': {
            20: r'scripts/sqp_vs_mpc_K_exp/10lap_sqp_nmpc_N20.npy',
            25: r'scripts/sqp_vs_mpc_K_exp/10lap_sqp_nmpc_N25.npy',
            30: r'scripts/sqp_vs_mpc_K_exp/10lap_sqp_nmpc_N30.npy'
        },
        'MPC': {
            20: r'scripts/sqp_vs_mpc_K_exp/10lap_mpc_N20.npy',
            25: r'scripts/sqp_vs_mpc_K_exp/10lap_mpc_N25.npy',
            30: r'scripts/sqp_vs_mpc_K_exp/10lap_mpc_N30.npy'
        }
    }

    # Execute the two comparison scripts sequentially
    
    # Run Comparison 1
    plot_controller_comparison_pid_mpc_sqp(pid_mpc_sqp_files)
    
    # Run Comparison 2
    plot_horizon_comparison_sqp_mpc(horizon_experiment_files)