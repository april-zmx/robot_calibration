# Robot Calibration Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable Python scaffold for multi-joint robot kinematics and dynamics calibration.

**Architecture:** Use a `src/robot_calibration` package with small modules for models, data, filtering, kinematics, dynamics, optimization, trajectory generation, metrics, and CLI commands. Keep the core independent from ROS and hardware so tests and examples run locally.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pytest, pyproject-based packaging.

---

## File Structure

- Create: `pyproject.toml` for packaging, dependencies, pytest configuration, and CLI entry point.
- Modify: `README.md` with quickstart, module map, and example commands.
- Create: `src/robot_calibration/__init__.py`
- Create: `src/robot_calibration/models/__init__.py`
- Create: `src/robot_calibration/models/robot.py`
- Create: `src/robot_calibration/data/__init__.py`
- Create: `src/robot_calibration/data/dataset.py`
- Create: `src/robot_calibration/data/loaders.py`
- Create: `src/robot_calibration/filtering/__init__.py`
- Create: `src/robot_calibration/filtering/butterworth.py`
- Create: `src/robot_calibration/filtering/differentiation.py`
- Create: `src/robot_calibration/filtering/pipeline.py`
- Create: `src/robot_calibration/kinematics/__init__.py`
- Create: `src/robot_calibration/kinematics/dh.py`
- Create: `src/robot_calibration/kinematics/calibration.py`
- Create: `src/robot_calibration/dynamics/__init__.py`
- Create: `src/robot_calibration/dynamics/friction.py`
- Create: `src/robot_calibration/dynamics/observation.py`
- Create: `src/robot_calibration/dynamics/base_parameters.py`
- Create: `src/robot_calibration/dynamics/identification.py`
- Create: `src/robot_calibration/dynamics/validation.py`
- Create: `src/robot_calibration/optimization/__init__.py`
- Create: `src/robot_calibration/optimization/least_squares.py`
- Create: `src/robot_calibration/trajectory/__init__.py`
- Create: `src/robot_calibration/trajectory/fourier.py`
- Create: `src/robot_calibration/metrics/__init__.py`
- Create: `src/robot_calibration/metrics/errors.py`
- Create: `src/robot_calibration/cli/__init__.py`
- Create: `src/robot_calibration/cli/main.py`
- Create: `examples/kinematics_2r.py`
- Create: `examples/dynamics_ols.py`
- Create: `tests/` unit tests matching the modules above.

---

### Task 1: Packaging And Public Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/robot_calibration/__init__.py`
- Modify: `README.md`
- Test: `tests/test_imports.py`

- [ ] **Step 1: Write failing import and CLI metadata tests**

```python
def test_package_imports():
    import robot_calibration
    assert robot_calibration.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_imports.py -v`
Expected: FAIL because package does not exist.

- [ ] **Step 3: Add package skeleton and pyproject**

Implement `pyproject.toml`, `src/robot_calibration/__init__.py`, and README quickstart.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_imports.py -v`
Expected: PASS.

---

### Task 2: Dataset Model And Loaders

**Files:**
- Create: `src/robot_calibration/data/dataset.py`
- Create: `src/robot_calibration/data/loaders.py`
- Create: `tests/test_dataset.py`
- Create: `tests/test_loaders.py`

- [ ] **Step 1: Write failing tests for dataset validation**

Cover monotonic time, shape mismatch rejection, joint count inference, and crop-by-index behavior.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset.py tests/test_loaders.py -v`
Expected: FAIL because data module is missing.

- [ ] **Step 3: Implement `CalibrationDataset` and CSV loader**

Use a frozen dataclass with arrays for `time`, `positions`, optional `velocities`, `accelerations`, `torques`, `currents`, and `metadata`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py tests/test_loaders.py -v`
Expected: PASS.

---

### Task 3: Filtering Pipeline

**Files:**
- Create: `src/robot_calibration/filtering/butterworth.py`
- Create: `src/robot_calibration/filtering/differentiation.py`
- Create: `src/robot_calibration/filtering/pipeline.py`
- Create: `tests/test_filtering.py`

- [ ] **Step 1: Write failing filtering tests**

Cover zero-phase low-pass shape preservation, noise reduction, central-difference derivative estimates, and pipeline immutability.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_filtering.py -v`
Expected: FAIL because filtering module is missing.

- [ ] **Step 3: Implement filtering functions**

Use `scipy.signal.butter`, `scipy.signal.filtfilt`, and central difference. Return copied datasets from the pipeline.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_filtering.py -v`
Expected: PASS.

---

### Task 4: Kinematics Calibration

**Files:**
- Create: `src/robot_calibration/models/robot.py`
- Create: `src/robot_calibration/kinematics/dh.py`
- Create: `src/robot_calibration/kinematics/calibration.py`
- Create: `tests/test_kinematics.py`

- [ ] **Step 1: Write failing DH and calibration tests**

Cover a known 2R planar chain and convergence from noisy synthetic position observations.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kinematics.py -v`
Expected: FAIL because kinematics module is missing.

- [ ] **Step 3: Implement DH transforms and least-squares calibration**

Use `scipy.optimize.least_squares` to estimate selected parameter offsets.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kinematics.py -v`
Expected: PASS.

---

### Task 5: Dynamics Identification Core

**Files:**
- Create: `src/robot_calibration/dynamics/friction.py`
- Create: `src/robot_calibration/dynamics/observation.py`
- Create: `src/robot_calibration/dynamics/base_parameters.py`
- Create: `src/robot_calibration/dynamics/identification.py`
- Create: `src/robot_calibration/dynamics/validation.py`
- Create: `tests/test_dynamics.py`

- [ ] **Step 1: Write failing dynamics tests**

Cover linear friction regressor layout, observation matrix assembly, QR base-parameter mapping, OLS recovery of known parameters, and torque prediction.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dynamics.py -v`
Expected: FAIL because dynamics module is missing.

- [ ] **Step 3: Implement dynamics core**

Implement protocol-based regressors, friction blocks, `numpy.linalg.lstsq`, pivoted QR via `scipy.linalg.qr`, and parameter statistics.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dynamics.py -v`
Expected: PASS.

---

### Task 6: Trajectory And Metrics

**Files:**
- Create: `src/robot_calibration/trajectory/fourier.py`
- Create: `src/robot_calibration/metrics/errors.py`
- Create: `tests/test_trajectory.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing trajectory and metric tests**

Cover Fourier trajectory shapes, derivatives, mixed trajectory boundary behavior, relative residual error, RMSE, and condition number.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory.py tests/test_metrics.py -v`
Expected: FAIL because modules are missing.

- [ ] **Step 3: Implement trajectory and metric utilities**

Implement truncated Fourier evaluation, fifth-order boundary correction, mixed trajectory evaluation, and metric helpers.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory.py tests/test_metrics.py -v`
Expected: PASS.

---

### Task 7: CLI And Examples

**Files:**
- Create: `src/robot_calibration/cli/main.py`
- Create: `examples/kinematics_2r.py`
- Create: `examples/dynamics_ols.py`
- Create: `tests/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI smoke tests**

Cover `robot-calib --help`, `robot-calib kinematics-demo`, and `robot-calib dynamics-demo`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL because CLI is missing.

- [ ] **Step 3: Implement CLI and examples**

Use `argparse`, call public APIs, and print compact calibration summaries.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS.

---

### Task 8: Full Verification

**Files:**
- Modify as needed only for fixes discovered by verification.

- [ ] **Step 1: Run full test suite**

Run: `pytest -v`
Expected: PASS.

- [ ] **Step 2: Run example scripts**

Run: `python examples/kinematics_2r.py`
Expected: Prints a small residual and estimated parameters.

Run: `python examples/dynamics_ols.py`
Expected: Prints recovered synthetic dynamic parameters and residual metrics.

- [ ] **Step 3: Run CLI smoke commands**

Run: `robot-calib --help`
Expected: Shows available subcommands.

- [ ] **Step 4: Review git diff**

Run: `git diff --stat`
Expected: Only planned files changed.
