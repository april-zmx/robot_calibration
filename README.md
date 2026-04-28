# robot_calibration

Python tools for multi-joint robot kinematics and dynamics calibration.

The first version of this repository is designed as a pure Python core
library: algorithms and data models stay independent from ROS, robot
vendors, and hardware drivers, while CLI commands and examples provide
simple runnable workflows.

Additional repository documentation:

- `docs/features.md`: current implemented capabilities and module overview
- `docs/roadmap.md`: optimization and refactor priorities
- `docs/piper_excitation_notes.md`: Piper excitation workflow, observation
  matrix definition, and current condition-number analysis

## Quickstart

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv .venv
UV_CACHE_DIR=/tmp/uv-cache uv pip install -e ".[dev]"
.venv/bin/pytest -v
.venv/bin/robot-calib --help
.venv/bin/robot-calib kinematics-demo
.venv/bin/robot-calib dynamics-demo
.venv/bin/robot-calib current-dynamics-demo
.venv/bin/robot-calib identify-ur-current path/to/ur.csv --drive-gains 14.87,13.26,11.13,10.62,11.03,11.47
```

## Planned Modules

- `models`: shared robot and calibration data structures.
- `data`: calibration dataset containers and CSV/JSON loaders.
- `filtering`: low-pass filtering, central differences, and preprocessing.
- `kinematics`: DH forward kinematics and least-squares calibration.
- `dynamics`: regressors, friction, base parameters, OLS identification.
- `trajectory`: Fourier and polynomial excitation trajectories.
- `metrics`: residual and conditioning metrics.
- `cli`: command-line demos and utilities.

## Examples

```bash
.venv/bin/python examples/kinematics_2r.py
.venv/bin/python examples/dynamics_ols.py
```

## Excitation Trajectory Generation

The trajectory module supports Fourier excitation trajectories, fifth-order
boundary correction, and `pymoo`-based condition-number optimization. The
optimizer searches Fourier sine/cosine coefficients, samples the resulting
trajectory, stacks a user-provided dynamics regressor, and minimizes the
effective condition number of the observation matrix.

For a URDF-backed workflow, install the optional Pinocchio dependency:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install -e ".[pinocchio]"
```

The repository includes a Piper example using AgileX's `agx_arm_urdf` layout.
If the URDF repository is not present yet:

```bash
git clone https://github.com/agilexrobotics/agx_arm_urdf.git agx_arm_urdf
```

Generate a bounded excitation trajectory and plot joint position, velocity, and
acceleration:

```bash
.venv/bin/python examples/piper_excitation_trajectory.py \
  --urdf agx_arm_urdf/piper/urdf/piper_description.urdf \
  --duration 4.0 \
  --sample-count 60 \
  --n-harmonics 2 \
  --max-evaluations 30 \
  --coefficient-bound 0.25 \
  --acceleration-limit 8.0 \
  --output-dir outputs/piper_excitation
```

The example enables motor reflected inertia columns by default. To generate and
label a rigid-body-only run instead, add:

```bash
  --no-motor-dynamics
```

Outputs:

- `outputs/piper_excitation/piper_excitation.npz`
- `outputs/piper_excitation/piper_excitation_summary.json`
- `outputs/piper_excitation/piper_excitation_plot.png`

The summary JSON now records both `include_motor_dynamics` and a compact
`dynamics_variant` label (`motor_on` or `motor_off`).

For notes on how the Piper observation matrix is constructed, what
`include_motor_dynamics=True` adds, and why large condition numbers can still
appear even when the matrix is full rank, see `docs/piper_excitation_notes.md`.

To visualize the generated trajectory in ROS2, use the system ROS Python after
sourcing Humble. The script publishes `/joint_states` and starts
`robot_state_publisher`; open `rviz2` separately and add `RobotModel` and `TF`.

```bash
source /opt/ros/humble/setup.bash

python3 examples/piper_rviz_visualization.py \
  --trajectory outputs/piper_excitation/piper_excitation.npz \
  --urdf agx_arm_urdf/piper/urdf/piper_description.urdf \
  --agx-root agx_arm_urdf
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
rviz2
```

On WSL, if RViz has OpenGL/D3D12 issues, start RViz with software rendering:

```bash
source /opt/ros/humble/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export QT_X11_NO_MITSHM=1
rviz2
```

## UR CSV Layout

`robot_calibration.data.loaders.load_ur_csv()` supports the no-header CSV layout
used by the reference UR10e MATLAB workflow:

```text
time,
q1..q6,
qd1..qd6,
i1..i6,
i_des1..i_des6,
tau_des1..tau_des6
```

The loader normalizes time so the selected window starts at zero and stores
desired currents and desired torques as first-class optional dataset signals.
For the exact filtering convention used in the reference UR10e notebook,
`robot_calibration.filtering.preprocess_reference_ur10e_dataset()` now mirrors
the notebook's Butterworth settings, central-difference acceleration estimate,
and `filtfilt(..., padtype=None)` behavior.

## Current-Driven Dynamics Identification

The current-driven workflow follows the Python notebook pattern from
`/home/april/robotics/dynamic_calibration/robot_dynamics.ipynb`:

```text
q, qd, qdd, motor_current
  -> motor_current * drive_gains
  -> Y(q, qd, qdd)
  -> optional QR base-parameter projection
  -> optional linear friction regressor
  -> ordinary least-squares identification
```

The core API is `robot_calibration.dynamics.identify_current_driven_dynamics`.
It accepts any callable regressor, so a future Pinocchio-backed regressor can be
plugged in without changing the identification workflow.

For the same `ur10e.urdf` and `ur-20_02_10-30sec_12harm.csv` assets used by the
reference notebook, the repository now also includes a tested Python baseline
that combines:

- `load_ur_csv(..., start_index=635, stop_index=3510)`
- `preprocess_reference_ur10e_dataset(...)`
- `PinocchioDynamicsModel.from_urdf(..., include_motor_dynamics=True)`
- `CurrentDrivenBaseDynamicsIdentifier(..., drive_gains=[14.87, 13.26, 11.13, 10.62, 11.03, 11.47])`

To generate a machine-readable comparison report against the saved notebook and
trajectory-optimization outputs, run:

```bash
source .venv/bin/activate
PYTHONPATH=src python examples/matlab_reference_consistency.py
```

For a UR-style current log, the CLI can run the same loading, filtering, torque
conversion, and OLS workflow:

```bash
.venv/bin/robot-calib identify-ur-current dataset.csv \
  --start-index 635 \
  --stop-index 3510 \
  --drive-gains 14.87,13.26,11.13,10.62,11.03,11.47
```

The current built-in CLI regressor is `acceleration-diagonal`, which is mainly a
lightweight smoke-test backend. Use `PinocchioRegressor` from Python for full
rigid-body regressors, or run the optional Pinocchio CLI backend:

```bash
.venv/bin/robot-calib identify-ur-current dataset.csv \
  --start-index 635 \
  --stop-index 3510 \
  --drive-gains 14.87,13.26,11.13,10.62,11.03,11.47 \
  --regressor pinocchio \
  --urdf /path/to/robot.urdf \
  --include-motor-dynamics
```

For simple physical constraints such as nonnegative mass, motor inertia, or
friction terms, use bounded least squares:

```bash
.venv/bin/robot-calib identify-ur-current dataset.csv \
  --drive-gains 14.87,13.26,11.13,10.62,11.03,11.47 \
  --method bounded \
  --nonnegative-indices 0,1,2
```

This first constrained path uses SciPy bounded least squares. Full inertial
matrix LMI constraints remain a future CVXPY/SDP extension.

## Optional Pinocchio Regressor

`robot_calibration.dynamics.PinocchioRegressor` wraps:

```python
pin.computeJointTorqueRegressor(model, data, q, qd, qdd)
```

and can optionally insert one motor reflected-inertia column after each link
parameter block, matching the notebook implementation. Install the optional
dependency only when you need that integration:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install -e ".[pinocchio]"
```

## Optional CVXPY Constraints

Install CVXPY support when you want to run the first physically constrained
least-squares path:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install -e ".[cvxpy]"
```

Then select the CVXPY method:

```bash
.venv/bin/robot-calib identify-ur-current dataset.csv \
  --drive-gains 14.87,13.26,11.13,10.62,11.03,11.47 \
  --method cvxpy \
  --nonnegative-indices 0,1,2
```

For Pinocchio-style dynamic parameter blocks, add pseudo-inertia semidefinite
constraints:

```bash
.venv/bin/robot-calib identify-ur-current dataset.csv \
  --drive-gains 14.87,13.26,11.13,10.62,11.03,11.47 \
  --method cvxpy \
  --regressor pinocchio \
  --urdf /path/to/robot.urdf \
  --include-motor-dynamics \
  --physical-consistency
```

The physical block layout is `[m, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz, Im?]`,
matching the Pinocchio notebook workflow.
