# Repository Features

## Overview

`robot_calibration` is a Python-first repository for multi-joint robot
kinematics and dynamics calibration. The current codebase is organized as a
core library under `src/robot_calibration`, with CLI entry points, examples,
and tests around it.

The repository is already usable as a calibration scaffold: it can load
trajectory data, preprocess signals, run synthetic kinematics calibration,
assemble dynamics regressors, estimate parameters, and execute smoke-test
workflows from the command line.

## Current Package Layout

```text
src/robot_calibration/
  cli/
  data/
  dynamics/
  filtering/
  kinematics/
  metrics/
  models/
  trajectory/
```

## Implemented Features

### 1. Data Models And Loaders

Files:

- `src/robot_calibration/data/dataset.py`
- `src/robot_calibration/data/loaders.py`

Current capabilities:

- `CalibrationDataset` stores:
  - `time`
  - `positions`
  - `velocities`
  - `accelerations`
  - `torques`
  - `currents`
  - `desired_currents`
  - `desired_torques`
- Validates shape consistency and monotonic timestamps.
- Supports dataset cropping by index range.
- Loads generic CSV data.
- Loads UR-style no-header CSV logs with the layout:

```text
time, q1..q6, qd1..q6, i1..i6, i_des1..i_des6, tau_des1..tau_des6
```

- Normalizes selected UR CSV windows so time starts at zero.

### 2. Signal Filtering And Preprocessing

Files:

- `src/robot_calibration/filtering/butterworth.py`
- `src/robot_calibration/filtering/differentiation.py`
- `src/robot_calibration/filtering/pipeline.py`

Current capabilities:

- Zero-phase Butterworth low-pass filtering using `scipy.signal.filtfilt`.
- Central-difference derivative estimation.
- Dataset preprocessing pipeline that:
  - filters joint velocities
  - estimates accelerations from filtered velocities
  - filters currents, desired currents, and desired torques
- Reference UR10e preprocessing helper
  `preprocess_reference_ur10e_dataset()` that matches the notebook/MATLAB
  settings:
  - normalized Butterworth cutoffs `0.15 / 0.2 / 0.15`
  - acceleration from filtered velocity with zeroed endpoints before filtering
  - `filtfilt(..., padtype=None)`
- Returns a new processed dataset instead of mutating inputs.

This follows the same broad preprocessing idea as the reference
`dynamic_calibration` workflow: clean the measured signals first, then build
identification inputs from filtered velocity and acceleration signals.

### 3. Kinematics

Files:

- `src/robot_calibration/kinematics/dh.py`
- `src/robot_calibration/kinematics/calibration.py`

Current capabilities:

- Standard DH parameter representation.
- Forward kinematics for serial chains.
- Least-squares calibration of selected DH offsets.
- Synthetic 2R calibration example in `examples/kinematics_2r.py`.

This gives the repository an end-to-end runnable kinematics path without
depending on robot hardware or ROS.

### 4. Dynamics Identification

Files:

- `src/robot_calibration/dynamics/observation.py`
- `src/robot_calibration/dynamics/friction.py`
- `src/robot_calibration/dynamics/base_parameters.py`
- `src/robot_calibration/dynamics/identification.py`
- `src/robot_calibration/dynamics/current_identification.py`
- `src/robot_calibration/dynamics/constrained_identification.py`
- `src/robot_calibration/dynamics/cvxpy_identification.py`
- `src/robot_calibration/dynamics/validation.py`

Current capabilities:

- Linear friction regressor with viscous, Coulomb, and offset terms.
- Observation matrix assembly across a trajectory.
- QR-based base-parameter extraction.
- Ordinary least squares parameter identification.
- Bounded least squares with simple lower/upper bounds.
- CVXPY-based constrained least squares with:
  - nonnegative parameter constraints
  - mass upper bounds
  - pseudo-inertia semidefinite constraints for physical consistency
- Torque prediction and residual validation utilities.

### 5. Current-Driven Dynamics Workflow

Files:

- `src/robot_calibration/dynamics/current_identification.py`

Current capabilities:

- Converts motor currents to joint torques through drive gains.
- Builds joint-torque observation matrices from `q`, `qd`, `qdd`.
- Optionally appends friction regressors.
- Supports three identification modes:
  - `ols`
  - `bounded`
  - `cvxpy`

This is the main bridge from logged current data to dynamic parameter
identification, and it mirrors the workflow used in the reference notebook.

### 6. Pinocchio Integration

Files:

- `src/robot_calibration/dynamics/pinocchio_adapter.py`
- `src/robot_calibration/dynamics/pinocchio_dynamics.py`

Current capabilities:

- Optional Pinocchio-backed torque regressor wrapper.
- URDF-based model construction helper.
- Model-only URDF construction helper for workflows that do not need mesh
  geometry loading.
- Optional motor reflected inertia columns.
- Notebook-style Pinocchio dynamics workflow helpers for:
  - dynamic parameter vector extraction
  - QR base-parameter extraction
  - current-driven base-parameter identification
- Tested reference UR10e workflow coverage using:
  - `URe/urdf/ur10e.urdf`
  - `dataset_ur10e/identification_data/ur-20_02_10-30sec_12harm.csv`
  - notebook drive gains `[14.87, 13.26, 11.13, 10.62, 11.03, 11.47]`

When motor dynamics are enabled, each joint contributes one extra reflected
motor inertia parameter `Im`, extending the standard Pinocchio-style block from
`10` parameters to `11`:

```text
[m, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz, Im]
```

This path is intentionally optional so the core package stays lightweight, while
still leaving a clean integration point for full rigid-body regressors.

### 7. Excitation Trajectory Generation And Optimization

Files:

- `src/robot_calibration/trajectory/fourier.py`
- `src/robot_calibration/trajectory/optimization.py`

Current capabilities:

- Function-style Fourier trajectory utilities for position, velocity, and
  acceleration sampling.
- Class-based trajectory APIs:
  - `FourierExcitationTrajectory`
  - `FifthOrderBoundaryPolynomial`
  - `MixedExcitationTrajectory`
  - `ExcitationTrajectoryGenerator`
- Fifth-order boundary correction so generated trajectories start and end at a
  requested rest configuration.
- `pymoo`-backed condition-number optimization:
  - `ExcitationConditionNumberProblem`
  - `ConditionNumberExcitationOptimizer`
  - `ExcitationOptimizationResult`
- Optional Fourier sine/cosine coefficient bounds.
- Joint position, velocity, and acceleration inequality constraints.
- Effective condition-number evaluation for structurally rank-deficient
  dynamics regressors.
- User-supplied regressor interface, including optional Pinocchio regressors
  built from URDF.

The optimizer searches Fourier coefficients, samples the mixed trajectory,
stacks the inverse-dynamics observation matrix, and minimizes the condition
number of that matrix. For the Piper workflow, the current implementation
matches the MATLAB reference structure:

```text
W = [Y(q, qd, qdd) * E1, Y_friction(qd)]
```

where `E1` is the QR-derived base-parameter projection. This means the reported
optimization rank is the rank of the augmented matrix `[Y * E1, Y_friction]`,
not just the pure base-dynamic block. This is the first complete
trajectory-design path in the repository.

### 8. Piper URDF Trajectory Example And ROS2 Visualization

Files:

- `examples/piper_excitation_trajectory.py`
- `examples/piper_rviz_visualization.py`
- `examples/matlab_reference_consistency.py`

Current capabilities:

- Parses revolute joint position and velocity limits from the AgileX Piper
  URDF.
- Builds a Pinocchio dynamics regressor from the Piper URDF without requiring
  mesh geometry loading.
- Generates an excitation trajectory for the Piper arm, with optional explicit
  Fourier coefficient bounds.
- Writes trajectory arrays, optimization metadata, and joint plots:
  - `piper_excitation.npz`
  - `piper_excitation_summary.json`
  - `piper_excitation_plot.png`
- Provides a ROS2 helper that:
  - rewrites Piper mesh paths to local file URIs for RViz compatibility
  - starts `robot_state_publisher`
  - publishes the generated trajectory on `/joint_states`
- Provides a reference-comparison helper that checks the Python UR10e
  identification and trajectory implementation against:
  - `robot_dynamics.ipynb`
  - `ptrnSrch_N12T30QR.mat`

The ROS2 helper intentionally keeps RViz setup manual: users open `rviz2`
themselves and add `RobotModel` and `TF` displays as needed.

For the current Piper-specific condition-number analysis and the verified
effects of motor inertia and coefficient bounds, see
`docs/piper_excitation_notes.md`.

### 9. Metrics

Files:

- `src/robot_calibration/metrics/errors.py`

Current capabilities:

- RMSE helpers.
- Relative residual error.
- Observation matrix condition number.

### 10. CLI Workflows

Files:

- `src/robot_calibration/cli/main.py`

Current commands:

- `robot-calib kinematics-demo`
- `robot-calib dynamics-demo`
- `robot-calib current-dynamics-demo`
- `robot-calib identify-ur-current`

The UR current identification CLI supports:

- start and stop index selection
- drive gain parsing
- optional filtering
- `acceleration-diagonal` smoke-test regressor
- `pinocchio` regressor backend
- `ols`, `bounded`, and `cvxpy` identification methods
- optional physical consistency constraints

## Examples

Available examples:

- `examples/kinematics_2r.py`
- `examples/dynamics_ols.py`
- `examples/piper_excitation_trajectory.py`
- `examples/piper_rviz_visualization.py`

The kinematics and OLS examples are intentionally small and synthetic. The
Piper examples exercise the URDF-backed trajectory generation path and optional
ROS2 visualization workflow.

## Test Coverage

Files under `tests/` currently cover:

- dataset validation and cropping
- CSV loading
- filtering and differentiation
- kinematics
- dynamics identification
- constrained and CVXPY-based identification
- trajectory utilities
- trajectory optimization
- metrics
- CLI smoke tests
- Pinocchio adapter behavior
- Piper excitation trajectory and RViz helper behavior
- reference UR data loading and preprocessing

## Current Limitations

The repository is functional, but there are some important gaps:

- The public API is still mixed between dataclasses and function-style helpers.
- Excitation trajectory optimization exists, but the ROS2 visualization path is
  still example-script based rather than a packaged ROS2 launch workflow.
- Class-based encapsulation is not yet consistent across modules.
- Base-parameter identification and physical consistency constraints are not yet
  fully unified in one end-to-end path.
- The current README is useful, but it is not the only place where users should
  discover repository capabilities; this document now fills that gap.

## Recommended Entry Points

For day-to-day use, start with:

- `README.md` for environment setup and quickstart commands
- `src/robot_calibration/cli/main.py` for runnable workflows
- `examples/` for minimal end-to-end examples
- `docs/roadmap.md` for upcoming refactors and optimization priorities
