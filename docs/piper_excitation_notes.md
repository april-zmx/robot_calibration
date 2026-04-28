# Piper Excitation Notes

## Scope

This note summarizes how the Piper excitation-trajectory workflow is currently
constructed in `robot_calibration`, and what has been verified so far about the
large observation-matrix condition numbers seen during trajectory optimization.

The implementation path described here is the one used by:

- `examples/piper_excitation_trajectory.py`
- `src/robot_calibration/trajectory/optimization.py`
- `src/robot_calibration/dynamics/pinocchio_dynamics.py`

It follows the same high-level structure as the reference MATLAB workflow in
`/home/april/robotics/dynamic_calibration/trajectory_optmzn`.

## Observation Matrix Definition

For trajectory optimization, the repository currently builds the same augmented
observation matrix used by the MATLAB reference cost function:

```text
W = [Y(q, qd, qdd) * E1, Y_friction(qd)]
```

where:

- `Y` is the full inverse-dynamics regressor.
- `E1` is the base-parameter projection matrix extracted from QR pivoting.
- `Y_friction` is the linear friction regressor with viscous, Coulomb, and
  offset terms.

This matches the reference MATLAB code in
`trajectory_optmzn/traj_cost_lgr.m`, which stacks:

```text
[regressorWithMotorDynamics(...) * E1, frictionRegressor(qd)]
```

for every sample.

### Piper Column Counts

For the current Piper URDF workflow:

- Number of joints: `6`
- Full dynamic parameters with motor dynamics: `66`
- Base dynamic parameters after QR: `41`
- Friction parameters: `18` (`3` per joint)

So the optimization-stage observation matrix has:

```text
41 + 18 = 59 columns
```

With `sample_count=200`, the resulting matrix shape is:

```text
W.shape = (1200, 59)
```

Therefore:

- `observation_rank = 59`
- `target_rank = 59`

means that the augmented matrix `[Y * E1, Y_friction]` is full column rank.
It does **not** mean that the pure base-dynamic part alone has `59` columns.

## Meaning Of `include_motor_dynamics=True`

`include_motor_dynamics=True` adds one extra parameter per joint for motor
reflected inertia.

Without motor dynamics, each joint contributes the standard `10` rigid-body
parameters:

```text
[m, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
```

With motor dynamics enabled, each joint contributes:

```text
[m, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz, Im]
```

where `Im` is the additional motor inertia parameter.

In the regressor, this is modeled exactly as in the notebook by inserting one
extra column proportional to joint acceleration:

```text
tau_motor_i = Im_i * qdd_i
```

So for Piper:

- `include_motor_dynamics=False` -> `6 * 10 = 60` dynamic parameters
- `include_motor_dynamics=True` -> `6 * 11 = 66` dynamic parameters

The friction model is still separate and adds `18` more columns only when the
augmented observation matrix is built.

## Verified Findings On Large Condition Numbers

The large condition numbers observed so far are not caused by an incorrect rank
calculation. The main verified findings are:

### 1. The QR base-parameter extraction is stable for Piper

Across different random seeds and QR sampling counts, the Piper base-dynamic
parameter count remained stable at `41`.

This makes it unlikely that the large condition number is caused by unstable
base-parameter extraction.

### 2. The matrix is full rank but still badly conditioned

For a representative Piper trajectory:

- `rank(W) = 59`
- `sigma_max(W) ~= 2.29e2`
- `sigma_min(W) ~= 1.95e-7`
- `cond(W) ~= 1.17e9`

So the issue is not rank deficiency. The issue is an extremely small smallest
singular value.

### 3. Friction is not the main source of the bad conditioning

For the same trajectory:

- `cond(Wb) ~= 1.06e9`, where `Wb = Y * E1`
- `cond(Wf) ~= 5.73`, where `Wf = Y_friction`
- `cond(W)  ~= 1.17e9`

This shows that most of the ill-conditioning is already present in the
base-dynamic part before friction columns are appended.

### 4. Motor dynamics are the dominant source of the jump in conditioning

Using the same Piper trajectory:

- Without motor dynamics:
  - `base_rank = 36`
  - `cond(Wb) ~= 80.7`
  - `cond(W)  ~= 124.1`
- With motor dynamics:
  - `base_rank = 41`
  - `cond(Wb) ~= 1.06e9`
  - `cond(W)  ~= 1.17e9`

This is the strongest current signal: enabling motor inertia identification is
what makes the observation matrix become severely ill-conditioned.

### 5. At least one motor-inertia direction is nearly collinear with a rigid-body inertia direction

For the current optimized Piper trajectory, one of the smallest-singular-value
directions is dominated by a pair of dynamic columns corresponding to:

- joint 2 `Izz`
- joint 2 `Im`

The raw regressor columns for that pair have correlation:

```text
0.9999999999999996
```

In practice this means that, on that trajectory, the optimizer cannot clearly
distinguish between part of the rigid-body inertia around the joint axis and
the motor reflected inertia.

### 6. Explicit coefficient bounds can matter, but removing them is not a complete fix

Earlier Piper experiments used:

```text
coefficient_bounds = (-0.5, 0.5)
```

and the optimized coefficients hit that bound.

In direct experiments with the same workflow and a moderate optimization
budget, relaxing the coefficient bound reduced the condition number:

- bound `0.5` -> `cond ~= 1.05e9`
- bound `1.0` -> `cond ~= 8.59e8`
- bound `2.0` -> `cond ~= 5.40e8`

This shows that a tight manual Fourier coefficient bound can limit excitation
quality. The MATLAB reference workflow relies primarily on trajectory
feasibility constraints and does not impose the same explicit coefficient box,
so the current Piper example now defaults to no explicit coefficient bound.

At the same time, simply removing the bound does not automatically solve the
conditioning problem. On Piper, unbounded runs still need enough optimization
budget, and the near-collinearity between some rigid-body inertia directions
and motor inertia directions remains the dominant issue.

## Current Interpretation

Based on the evidence above, the current large Piper condition numbers are most
likely caused by a combination of:

1. Motor inertia parameters being difficult to distinguish from some
   joint-axis rigid-body inertia terms.
2. Explicit Fourier coefficient bounds limiting how aggressively the optimizer
   can excite those directions.
3. Optimization budget and local-search settings not yet being fully aligned
   with the original MATLAB `patternsearch` setup.

## Practical Reading Of The Current Output

If the Piper excitation summary reports:

```text
observation rank: 59/59
condition number: 1e9 order
```

the correct interpretation is:

- the augmented matrix `[Y * E1, Y_friction]` is full column rank
- but some parameter directions, especially motor-inertia-related ones, are
  still very weakly excited

So full rank should not be interpreted as "easy identification" by itself.

## Recommended Next Steps

The most useful next investigations are:

1. Re-run the Piper optimization with relaxed or removed coefficient bounds
   while keeping the joint, velocity, and acceleration constraints.
2. Continue aligning `pymoo` pattern-search settings with the MATLAB
   `patternsearch` configuration.
3. If needed, treat motor inertia as a separate identification subproblem or
   add trajectory design targets that explicitly separate joint-axis inertia
   from reflected motor inertia.
