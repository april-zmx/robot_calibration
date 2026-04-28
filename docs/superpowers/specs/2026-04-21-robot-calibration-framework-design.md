# Robot Calibration Framework Design

## Goal

Build a Python-first repository for multi-joint robot kinematics and dynamics calibration. The first version should provide a clean, testable core library with command-line and example entry points, while keeping the algorithm layer independent from ROS, robot vendors, and data collection systems.

## Reference Algorithms

The design references the MATLAB repository at `/home/april/robotics/dynamic_calibration`, which implements UR10e dynamic calibration. Its useful algorithmic structure is:

- Generate inverse dynamics regressors from a URDF-derived rigid-body model.
- Express inverse dynamics in linear-in-parameters form: `tau = Y(q, qd, qdd) pi`.
- Add motor reflected inertia terms to the rigid-body regressor when needed.
- Use pivoted QR decomposition on stacked observation matrices to find identifiable base parameters.
- Use linear friction regressors with viscous, Coulomb, and offset terms.
- Preprocess measured trajectories with zero-phase low-pass filtering and central-difference acceleration estimates.
- Optimize excitation trajectories by minimizing the observation matrix condition number under joint, velocity, and acceleration limits.
- Estimate drive gains with unloaded and loaded trajectories when only motor currents are available.
- Estimate dynamic parameters with ordinary least squares first, then support physically constrained least squares later.
- Validate estimated parameters by comparing measured and predicted torques and reporting relative residual error per joint.

This repository should reuse those ideas, but expose them as small Python modules instead of porting the full MATLAB workflow directly.

## Architecture

The package will be named `robot_calibration`. It will be organized around explicit data models and narrow algorithm modules:

```text
robot_calibration/
  models/
  data/
  filtering/
  kinematics/
  dynamics/
  optimization/
  trajectory/
  metrics/
  cli/
```

The core package should depend on `numpy` and `scipy`. Optional future integrations such as ROS2, URDF parsers, Pinocchio, RBDL, or CVXPY should stay outside the first required path unless a specific feature needs them.

## Modules

### `models`

Defines shared robot and calibration data structures:

- Joint limits and joint state arrays.
- Denavit-Hartenberg parameters for simple kinematics examples.
- Inertial parameter vectors for dynamics identification.
- Calibration result containers with estimates, residuals, covariance, and metadata.

These models should be lightweight Python dataclasses and should validate array shapes at construction time where practical.

### `data`

Handles measured and synthetic calibration data:

- Load CSV and JSON trajectory data.
- Store time, positions, velocities, accelerations, torques, currents, and optional desired signals.
- Crop data by index or time interval.
- Validate monotonic timestamps and consistent sample counts.

The first version should support generic column names rather than only UR-specific layouts.

### `filtering`

Owns all signal preprocessing:

- Butterworth low-pass filtering with `scipy.signal.butter` and `scipy.signal.filtfilt`.
- Central-difference velocity and acceleration estimation.
- Moving average and Savitzky-Golay helpers for lighter workflows.
- Optional outlier detection using threshold or median absolute deviation.
- A composable preprocessing pipeline that returns a new dataset instead of mutating input data.

This directly reflects the `filterData.m` pattern from the reference repository: filter velocity/current, estimate acceleration by central difference, then zero-phase filter acceleration.

### `kinematics`

Provides first-version kinematic calibration:

- Forward kinematics for serial chains using DH parameters.
- Residual functions comparing predicted end-effector poses or positions against observations.
- Least-squares calibration of selected DH parameter offsets.

The first version should include a simple 2R or 3R synthetic example so the full calibration loop is runnable without hardware.

### `dynamics`

Provides dynamic identification building blocks:

- Regressor protocol: a callable interface that maps `q`, `qd`, `qdd` to `Y`.
- Observation matrix assembly across a trajectory.
- QR base-parameter extraction from stacked regressors.
- Linear friction regressor: `Fv * qd + Fc * sign(qd) + F0`.
- Ordinary least-squares dynamic parameter estimation.
- Parameter statistics: residual variance, covariance, standard deviation, and relative standard deviation.
- Torque prediction and validation helpers.

The first version should implement OLS and base-parameter utilities. Physically constrained least squares and drive-gain identification should be designed as extension points but not required in the first working path.

### `optimization`

Wraps numerical optimization routines:

- `scipy.optimize.least_squares` for nonlinear kinematics calibration.
- Linear least-squares helpers for dynamics.
- Shared result normalization so CLI and examples can print consistent summaries.

### `trajectory`

Provides calibration trajectory generation and future experiment design:

- Truncated Fourier-series trajectories.
- Fifth-order polynomial boundary correction.
- Mixed Fourier-plus-polynomial trajectories.
- Condition-number objective for excitation quality.

Full constrained trajectory optimization can be added later. The first version should generate and evaluate trajectories, then leave optimizer integration as a documented extension.

### `metrics`

Contains reusable evaluation metrics:

- Relative residual error per joint.
- Root-mean-square error.
- Condition number of an observation matrix.

### `cli`

Provides simple command-line workflows:

- Run a kinematics calibration example.
- Run a dynamics OLS example on synthetic or CSV data.
- Apply filtering to a dataset and write a processed CSV.

The CLI should exercise the same public APIs as user code and examples.

## Data Flow

The intended dynamics workflow is:

```text
raw csv/json
  -> data loader and validation
  -> filtering pipeline
  -> regressor and friction observation matrix assembly
  -> OLS or constrained identification
  -> torque prediction
  -> validation metrics and result export
```

The intended kinematics workflow is:

```text
robot model + pose observations
  -> forward kinematics
  -> residual construction
  -> nonlinear least squares
  -> calibrated model and residual report
```

## First-Version Scope

The first version should include:

- Python packaging with `pyproject.toml`.
- A clean `src/robot_calibration` package layout.
- Unit tests using `pytest`.
- Data model and loader support for generic calibration datasets.
- Filtering functions and pipeline.
- DH forward kinematics and synthetic kinematics calibration.
- Dynamics regressor protocol, observation matrix assembly, friction regressor, QR base-parameter extraction, OLS identification, and validation metrics.
- Fourier and mixed trajectory generation utilities.
- CLI smoke commands and examples.
- README with quickstart commands and module overview.

The first version should not require ROS, MATLAB, YALMIP, SDPT3, robot hardware, or vendor-specific drivers.

## Future Extensions

Planned extensions include:

- URDF import and robot-model conversion.
- Pinocchio or RBDL-backed inverse dynamics regressors.
- CVXPY-based physical consistency constraints.
- Drive-gain identification from loaded and unloaded experiments.
- ROS2 bag loaders and live data collection utilities.
- Richer trajectory optimization using condition-number or D-optimality objectives.

## Error Handling

The core APIs should raise clear `ValueError` exceptions for invalid shapes, missing required fields, non-monotonic timestamps, or incompatible joint counts. CLI commands should catch those errors and print concise user-facing messages.

Filtering should reject signals that are too short for the selected zero-phase filter settings. Least-squares routines should report rank and conditioning when available, because poorly excited trajectories can produce unstable estimates.

## Testing Strategy

Tests should cover:

- Dataset shape validation and cropping.
- Filtering preserves shape and reduces synthetic high-frequency noise.
- Central difference returns expected derivatives for polynomial signals.
- DH forward kinematics for known simple chains.
- Kinematics calibration converges on synthetic noisy observations.
- Friction regressor block structure.
- Observation matrix assembly shape and ordering.
- QR base-parameter extraction reproduces full-regressor torques on synthetic rank-deficient matrices.
- OLS recovers known parameters from synthetic data.
- Relative residual error and condition-number metrics.
- CLI smoke tests.

## Open Design Choices

- The first regressor implementation can be synthetic and protocol-based; production rigid-body regressors should be plugged in later.
- Physical consistency constraints should be a future optional dependency, most likely through CVXPY.
- ROS2 should remain an integration layer, not a dependency of the core package.
