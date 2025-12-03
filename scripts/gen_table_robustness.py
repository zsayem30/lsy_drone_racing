import numpy as np
import pandas as pd
from typing import List, Dict, Any

# --- Modified Performance Table Generation Function ---

def generate_performance_table(all_runs_data: List[Dict[str, Any]], solver_name: str, disturbance_scale: float) -> pd.DataFrame:
    """
    Extracts key performance metrics from a list of lap results, converts solver time
    from milliseconds to seconds, and creates a DataFrame with Solver and Disturbance 
    Scale in the index.

    Args:
        all_runs_data: A list where each element is a dictionary containing 
                       performance metrics for one lap/run.
        solver_name: A string identifying the controller type (e.g., 'SQP').
        disturbance_scale: The disturbance scale used in this experiment (e.g., 0.001).

    Returns:
        A Pandas DataFrame summarizing the performance.
    """
    
    table_data = []
    for i, run_data in enumerate(all_runs_data):
        # Retrieve avg solver time in milliseconds, convert to seconds
        solver_time_ms = run_data.get('avg_solver_time_ms', np.nan)
        solver_time_s = solver_time_ms / 1000.0 if not pd.isna(solver_time_ms) else np.nan
        
        lap_data = {
            "Solver": solver_name,
            # Renamed column to reflect the experiment variable
            "Disturbance Scale": disturbance_scale, 
            "Lap": i + 1,
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
        }
        table_data.append(lap_data)

    df = pd.DataFrame(table_data)
    
    # Set the MultiIndex for comparison
    df = df.set_index(["Solver", "Disturbance Scale", "Lap"])
    
    return df

# --- Execution for Comparison ---




if __name__ == "__main__":
    
    # New experiment file structure for robustness testing
    experiment_files = {
        'RTI SQP NMPC': {
            0.001: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/RTI_SQP_NMPC_noise_0_001.npy',
            0.005: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/RTI_SQP_NMPC_noise_0_005.npy',
            0.01: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/RTI_SQP_NMPC_noise_0_01.npy'
        },
        'SQP NMPC': {
            0.001: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/SQP_NMPC_noise_0_001.npy',
            0.005: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/SQP_MPC_noise_0_005.npy',
            0.01: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/SQP_MPC_noise_0_01.npy'
        },
        'PID': {
            0.001: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/PID_noise_0_001.npy',
            0.005: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/PID_noise_0_005.npy',
            0.01: r'/content/repos/lsy_drone_racing/lsy_drone_racing/experiments/PID_noise_0_01.npy'
        }
    }

    all_performance_dfs = []
    
    # Updated print statement to reflect the new variable
    print("## Loading and Processing Data by Disturbance Scale")
    
    # Iterate through the nested dictionary structure
    for solver_name, scale_data in experiment_files.items():
        # Renamed variable to 'scale' for clarity
        for scale_value, file_path in scale_data.items():
            try:
                run_data = np.load(file_path, allow_pickle=True).tolist()
                
                if run_data:
                    # Pass the scale_value to the function
                    df = generate_performance_table(run_data, solver_name, scale_value)
                    all_performance_dfs.append(df)
                    # Updated print statement
                    print(f"- Processed **{len(run_data)}** runs for **{solver_name} (Scale={scale_value})**.")
                else:
                    print(f"- Data for **{solver_name} (Scale={scale_value})** is empty and skipped.")

            except FileNotFoundError:
                print(f"- **Error:** File not found at `{file_path}`. Skipping.")
            except Exception as e:
                print(f"- **Error:** Unexpected error loading data for {solver_name} (Scale={scale_value}): {e}")
            
    print("-" * 60)
    
    if all_performance_dfs:
        # 2. Combine all DataFrames into a single raw comparison table
        raw_comparison_df = pd.concat(all_performance_dfs)

        # 3. Create the Aggregated Summary DataFrame (Successful Runs)
        # Group by both Solver AND Disturbance Scale
        summary_df = raw_comparison_df[raw_comparison_df['Success'] == True].groupby(['Solver', 'Disturbance Scale']).agg({
            "Lap Time [s]": ['mean', 'min', 'count'],
            "RMSE Tracking Error": 'mean',
            "Solver Time [s]": 'mean'
        })
        
        # Flatten the MultiIndex columns
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
        
        # Rename the columns to be clean metric row labels
        summary_df = summary_df.rename(columns={
            'Lap Time [s]_count': 'Successful Runs',
            'Lap Time [s]_mean': 'Avg Lap Time [s]',
            'Lap Time [s]_min': 'Best Lap Time [s]',
            'RMSE Tracking Error_mean': 'Avg RMSE Tracking Error',
            'Solver Time [s]_mean': 'Avg Solver Time [s]'
        })

        # 4. Transpose the DataFrame and use MultiIndex for columns
        transposed_summary_df = summary_df.T
        
        # 5. Format the numerical values in the transposed table
        for idx in transposed_summary_df.index:
            for col in transposed_summary_df.columns:
                value = transposed_summary_df.loc[idx, col]
                if 'Time' in idx or 'RMSE' in idx:
                     # Time and RMSE formatted to 4 decimals
                     transposed_summary_df.loc[idx, col] = f"{value:.4f}"
                elif 'Successful Runs' in idx:
                     # Count is formatted as an integer
                     transposed_summary_df.loc[idx, col] = f"{int(round(value))}"

        # 6. Print the resulting comparison table
        print("\n## Robustness Comparison by Disturbance Scale")
        print("### Metrics (Rows) vs. (Solver, Disturbance Scale) (Columns)")
        print("-" * 100)
        
        # Use to_markdown for a clean output
        print(transposed_summary_df.to_markdown(numalign="left", stralign="left"))
        
        print("-" * 100)
        
    else:
        print("No data was successfully loaded or processed. Cannot generate a comparison table.")


