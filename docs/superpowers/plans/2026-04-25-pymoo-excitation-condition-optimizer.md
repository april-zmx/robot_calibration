# Pymoo Excitation Condition Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `pymoo`-backed optimizer that searches Fourier excitation coefficients by minimizing the condition number of a stacked dynamics observation matrix.

**Architecture:** Create a focused trajectory optimization module that converts flat decision vectors into sine/cosine coefficient matrices, builds `MixedExcitationTrajectory` objects, samples `q/qd/qdd`, stacks a user-provided inverse-dynamics regressor, and exposes the objective as a `pymoo.core.problem.Problem`. A small optimizer wrapper runs `pymoo.optimize.minimize()` with `PatternSearch` by default.

**Tech Stack:** Python 3.10+, NumPy, pymoo, pytest.

---

## File Structure

- Modify: `pyproject.toml`
  - Add `pymoo>=0.6.1` as a direct dependency.
- Create: `src/robot_calibration/trajectory/optimization.py`
  - Add coefficient vector packing/unpacking helpers.
  - Add `ExcitationConditionNumberProblem`.
  - Add `ExcitationOptimizationResult`.
  - Add `ConditionNumberExcitationOptimizer`.
- Modify: `src/robot_calibration/trajectory/__init__.py`
  - Export the optimizer API.
- Create: `tests/test_trajectory_optimization.py`
  - Cover vector decoding, objective evaluation, solver wiring, and invalid bounds.

---

### Task 1: Dependency And Objective Problem

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_trajectory_optimization.py`
- Create: `src/robot_calibration/trajectory/optimization.py`
- Modify: `src/robot_calibration/trajectory/__init__.py`

- [ ] **Step 1: Write failing tests**

Test that:

- `ExcitationConditionNumberProblem` imports from `robot_calibration.trajectory`.
- `n_var == 2 * n_joints * n_harmonics`.
- Evaluating a zero decision vector returns a finite objective or a finite penalty.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_trajectory_optimization.py -v`

Expected: FAIL because the optimization module does not exist.

- [ ] **Step 3: Add dependency and minimal problem implementation**

Add `pymoo>=0.6.1` to `pyproject.toml`.

Implement `ExcitationConditionNumberProblem` as a `pymoo.core.problem.Problem` with a vectorized `_evaluate()` that loops over candidate rows and writes `out["F"]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_trajectory_optimization.py -v`

Expected: PASS after installing project dependencies.

---

### Task 2: Optimizer Wrapper

**Files:**
- Modify: `tests/test_trajectory_optimization.py`
- Modify: `src/robot_calibration/trajectory/optimization.py`
- Modify: `src/robot_calibration/trajectory/__init__.py`

- [ ] **Step 1: Write failing optimizer test**

Test that `ConditionNumberExcitationOptimizer.optimize()` returns:

- a `MixedExcitationTrajectory`
- `sine_coefficients` and `cosine_coefficients`
- a finite `condition_number`
- a sampled `time/q/qd/qdd` shape consistent with requested `sample_count`

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_trajectory_optimization.py::test_condition_number_optimizer_returns_trajectory_result -v`

Expected: FAIL because the wrapper does not exist.

- [ ] **Step 3: Implement optimizer wrapper**

Use `pymoo.algorithms.soo.nonconvex.pattern.PatternSearch` by default and call `pymoo.optimize.minimize(problem, algorithm, termination=("n_eval", max_evaluations), seed=seed, verbose=False)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_trajectory_optimization.py -v`

Expected: PASS.

---

### Task 3: Validation And Full Regression

**Files:**
- Modify: `tests/test_trajectory_optimization.py`
- Modify: `src/robot_calibration/trajectory/optimization.py`

- [ ] **Step 1: Write validation tests**

Cover:

- invalid coefficient bounds shape
- invalid `sample_count`
- regressor output with wrong row count

- [ ] **Step 2: Run validation tests**

Run: `.venv/bin/pytest tests/test_trajectory_optimization.py -v`

Expected: PASS.

- [ ] **Step 3: Run full suite**

Run: `.venv/bin/pytest -v`

Expected: PASS.

---

## Notes

- Keep the first optimizer single-objective only.
- Keep constraints out of this first implementation; joint, velocity, and acceleration limits can be added as `pymoo` inequality constraints in the next slice.
- Keep the regressor as a user-supplied callable so this optimizer works with synthetic tests and future Pinocchio regressors.
