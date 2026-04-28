# Repository Roadmap

## Purpose

This roadmap describes the next optimization and refactor steps for
`robot_calibration`. It is intentionally practical: the goal is to turn the
current working scaffold into a cleaner, more extensible calibration framework
without destabilizing the pieces that already run.

## Guiding Priorities

The near-term priorities are:

1. Prioritize excitation trajectory generation.
2. Move the main workflows toward class-based interfaces.
3. Keep the current runnable calibration paths intact while refactoring.
4. Avoid adding an unnecessary number of optimization backends.

## Stage 1: Documentation And API Stabilization

Goal:

- Make the current repository state easy to understand without reading chat
  history or inferring intent from tests.

Tasks:

- Keep `README.md` focused on quickstart and common commands.
- Maintain `docs/features.md` as the repository capability snapshot.
- Maintain this roadmap as the short planning document.
- Clarify which APIs are stable and which are still evolving.

Why first:

- The project has grown quickly, and a short written map reduces confusion
  before further refactoring begins.

## Stage 2: Excitation Trajectory Classes

Goal:

- Turn the current trajectory helper functions into explicit, reusable objects.

Target classes:

- `FourierExcitationTrajectory`
- `FifthOrderBoundaryPolynomial`
- `MixedExcitationTrajectory`
- `ExcitationTrajectoryGenerator`

Expected behavior:

- Represent excitation trajectory parameters explicitly.
- Sample `q`, `qd`, and `qdd` from one object.
- Cleanly serialize or print trajectory parameters for experiments.
- Make trajectory generation easier to pass into downstream identification code.

Notes:

- Existing function-level utilities can remain as internal numerical helpers or
  compatibility wrappers during the transition.

Status:

- Implemented. The package now exposes class-based Fourier, boundary
  polynomial, mixed trajectory, and generator APIs while preserving the original
  function helpers.

## Stage 3: Excitation Trajectory Optimizer

Goal:

- Add a complete excitation trajectory optimization path.

Planned design:

- A small optimizer API centered on one backend: `pymoo`.
- A pluggable objective interface.
- Default objective: observation matrix condition number minimization.
- Constraint support for:
  - joint position limits
  - joint velocity limits
  - joint acceleration limits

Why this shape:

- The repository already has the mathematical building blocks for excitation
  trajectories.
- `pymoo` is a reasonable fit for continuous nonlinear constrained search.
- Keeping one optimizer path avoids turning the project into an algorithm zoo
  too early.

Deferred for later:

- multiple optimizer backends
- large multi-objective frameworks
- advanced experiment design UI

Status:

- Implemented as a first working path. The `pymoo` optimizer minimizes the
  effective condition number of a stacked observation matrix and supports joint
  position, velocity, and acceleration constraints.
- A Piper URDF example now demonstrates URDF-backed trajectory generation,
  plotting, and ROS2 joint-state playback for RViz inspection.

Remaining improvements:

- Add a reusable CLI command for trajectory generation instead of only example
  scripts.
- Add optional export formats such as CSV and ROS bag.
- Add richer objective choices such as D-optimality or base-parameter projected
  condition number.
- Add a packaged ROS2 launch workflow for users who want one-command RViz
  visualization.

## Stage 4: Class-Based Workflow Wrappers

Goal:

- Wrap the main operational flows in classes so the public API becomes easier to
  discover and compose.

Target classes:

- `URCsvLoader`
- `SignalPreprocessor`
- `CurrentDrivenDynamicsIdentifier`
- `PinocchioRegressorBuilder`
- `ExcitationTrajectoryDesigner`

Expected outcome:

- CLI code can instantiate workflow objects instead of assembling several
  unrelated functions.
- Examples become easier to read.
- Module boundaries become clearer for future contributors.

Notes:

- The current function-based APIs do not need to disappear immediately.
- Backward-compatible wrappers are acceptable during the refactor.

## Stage 5: Stronger Dynamics Integration

Goal:

- Make the dynamics path more representative of real rigid-body identification
  while preserving the lightweight pure-Python core.

Next improvements:

- Strengthen the Pinocchio-backed path with more realistic end-to-end examples.
- Improve documentation around inertial parameter layouts.
- Add a cleaner bridge between base-parameter identification and full-parameter
  physical consistency constraints.
- Expand validation reports for measured-vs-predicted torque comparisons.
- Use optimized excitation trajectories directly in synthetic and real-data
  dynamics identification examples.

## Stage 6: Real-Data Workflows

Goal:

- Reduce the gap between synthetic demos and actual robot datasets.

Next improvements:

- Add more reference-data smoke tests.
- Add example commands and scripts for the reference UR dataset.
- Improve export and reporting for identified parameters.
- Consider dataset adapters for additional robot log formats beyond the current
  UR CSV layout.

## Longer-Term Extensions

Potential future work after the stages above:

- drive-gain identification from loaded and unloaded experiments
- richer physical consistency constraints
- ROS2 integration layers
- URDF-centric project templates
- notebook companions for research workflows
- packaged RViz launch files for generated excitation trajectories

## What Is Intentionally Not Prioritized Right Now

To keep momentum, the following are not first-line goals:

- a large menu of optimization algorithms
- early heavy ROS coupling
- vendor-specific data collection tooling in the core package
- broad refactors that do not support trajectory generation or workflow clarity

## Suggested Working Order

The practical next implementation sequence is:

1. package the trajectory optimizer as a CLI command
2. add export formats for generated trajectories
3. class wrappers for data/filtering/dynamics/trajectory workflows
4. stronger real-data and Pinocchio examples

This order keeps the repository aligned with the current product direction while
building on top of code that already works.
