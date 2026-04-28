# Excitation Trajectory Classes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable class-based excitation trajectory APIs while preserving the existing function-based trajectory helpers.

**Architecture:** Keep the numerical formulas in `src/robot_calibration/trajectory/fourier.py` and layer small trajectory objects on top of the existing helpers. The classes expose `sample(time)` and uniform sampling methods so downstream optimization and identification code can consume a single object interface.

**Tech Stack:** Python 3.10+, NumPy, pytest.

---

## File Structure

- Modify: `src/robot_calibration/trajectory/fourier.py`
  - Add `FourierExcitationTrajectory`.
  - Add `FifthOrderBoundaryPolynomial`.
  - Add `MixedExcitationTrajectory`.
  - Add `ExcitationTrajectoryGenerator`.
  - Reuse `fourier_series()`, `fifth_order_boundary_coefficients()`, and `mixed_trajectory()`.
- Modify: `src/robot_calibration/trajectory/__init__.py`
  - Export the new public classes.
- Modify: `tests/test_trajectory.py`
  - Add focused tests for class-based sampling, boundary behavior, uniform sampling, and validation.

---

### Task 1: Fourier Trajectory Object

**Files:**
- Modify: `tests/test_trajectory.py`
- Modify: `src/robot_calibration/trajectory/fourier.py`
- Modify: `src/robot_calibration/trajectory/__init__.py`

- [ ] **Step 1: Write the failing Fourier object test**

Add this import:

```python
from robot_calibration.trajectory import FourierExcitationTrajectory
```

Add this test:

```python
def test_fourier_excitation_trajectory_matches_function_helper():
    time = np.linspace(0.0, 1.0, 6)
    q0 = np.array([0.2, -0.4])
    a = np.array([[0.1, -0.2], [0.3, 0.05]])
    b = np.array([[0.05, 0.02], [-0.1, 0.04]])
    omega = 2.0 * np.pi
    trajectory = FourierExcitationTrajectory(q0, a, b, omega)

    q, qd, qdd = trajectory.sample(time)
    expected_q, expected_qd, expected_qdd = fourier_series(time, q0, a, b, omega)

    np.testing.assert_allclose(q, expected_q)
    np.testing.assert_allclose(qd, expected_qd)
    np.testing.assert_allclose(qdd, expected_qdd)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_trajectory.py::test_fourier_excitation_trajectory_matches_function_helper -v`

Expected: FAIL with an import error because `FourierExcitationTrajectory` is not exported yet.

- [ ] **Step 3: Implement the minimal Fourier object**

In `src/robot_calibration/trajectory/fourier.py`, add:

```python
class FourierExcitationTrajectory:
    """Truncated Fourier excitation trajectory."""

    def __init__(
        self,
        q0: ArrayLike,
        sine_coefficients: ArrayLike,
        cosine_coefficients: ArrayLike,
        fundamental_frequency: float,
    ) -> None:
        self.q0 = np.asarray(q0, dtype=float)
        self.sine_coefficients, self.cosine_coefficients = _coefficient_arrays(
            sine_coefficients,
            cosine_coefficients,
        )
        if self.q0.shape != (self.sine_coefficients.shape[0],):
            raise ValueError("q0 must have one entry per joint")
        if fundamental_frequency <= 0.0:
            raise ValueError("fundamental_frequency must be positive")
        self.fundamental_frequency = float(fundamental_frequency)

    @property
    def n_joints(self) -> int:
        return self.sine_coefficients.shape[0]

    @property
    def n_harmonics(self) -> int:
        return self.sine_coefficients.shape[1]

    def sample(
        self,
        time: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return fourier_series(
            time,
            self.q0,
            self.sine_coefficients,
            self.cosine_coefficients,
            self.fundamental_frequency,
        )
```

Export it from `src/robot_calibration/trajectory/__init__.py`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_trajectory.py::test_fourier_excitation_trajectory_matches_function_helper -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_trajectory.py src/robot_calibration/trajectory/fourier.py src/robot_calibration/trajectory/__init__.py
git commit -m "feat: add Fourier excitation trajectory object"
```

---

### Task 2: Boundary Polynomial Object

**Files:**
- Modify: `tests/test_trajectory.py`
- Modify: `src/robot_calibration/trajectory/fourier.py`
- Modify: `src/robot_calibration/trajectory/__init__.py`

- [ ] **Step 1: Write the failing boundary polynomial test**

Add this import:

```python
from robot_calibration.trajectory import FifthOrderBoundaryPolynomial
```

Add this test:

```python
def test_boundary_polynomial_samples_position_velocity_and_acceleration():
    coeffs = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    time = np.array([0.0, 0.5])
    polynomial = FifthOrderBoundaryPolynomial(coeffs)

    q, qd, qdd = polynomial.sample(time)

    expected_q = coeffs @ np.vstack([time**i for i in range(6)])
    expected_qd = (
        coeffs[:, 1:2]
        + 2.0 * coeffs[:, 2:3] * time
        + 3.0 * coeffs[:, 3:4] * time**2
        + 4.0 * coeffs[:, 4:5] * time**3
        + 5.0 * coeffs[:, 5:6] * time**4
    )
    expected_qdd = (
        2.0 * coeffs[:, 2:3]
        + 6.0 * coeffs[:, 3:4] * time
        + 12.0 * coeffs[:, 4:5] * time**2
        + 20.0 * coeffs[:, 5:6] * time**3
    )
    np.testing.assert_allclose(q, expected_q)
    np.testing.assert_allclose(qd, expected_qd)
    np.testing.assert_allclose(qdd, expected_qdd)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_trajectory.py::test_boundary_polynomial_samples_position_velocity_and_acceleration -v`

Expected: FAIL with an import error because `FifthOrderBoundaryPolynomial` is not exported yet.

- [ ] **Step 3: Implement the minimal polynomial object**

In `src/robot_calibration/trajectory/fourier.py`, add:

```python
class FifthOrderBoundaryPolynomial:
    """Fifth-order polynomial used to enforce trajectory boundary conditions."""

    def __init__(self, coefficients: ArrayLike) -> None:
        coeffs = np.asarray(coefficients, dtype=float)
        if coeffs.ndim != 2 or coeffs.shape[1] != 6:
            raise ValueError("coefficients must have shape (n_joints, 6)")
        self.coefficients = coeffs

    @classmethod
    def from_fourier_boundary(
        cls,
        duration: float,
        q0: ArrayLike,
        sine_coefficients: ArrayLike,
        cosine_coefficients: ArrayLike,
        fundamental_frequency: float,
    ) -> "FifthOrderBoundaryPolynomial":
        return cls(
            fifth_order_boundary_coefficients(
                duration,
                sine_coefficients,
                cosine_coefficients,
                fundamental_frequency,
                q0,
            )
        )

    @property
    def n_joints(self) -> int:
        return self.coefficients.shape[0]

    def sample(
        self,
        time: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        t = np.asarray(time, dtype=float)
        if t.ndim != 1:
            raise ValueError("time must be 1D")
        powers = np.vstack([t**i for i in range(6)])
        q = self.coefficients @ powers
        qd = (
            self.coefficients[:, 1:2]
            + 2.0 * self.coefficients[:, 2:3] * t
            + 3.0 * self.coefficients[:, 3:4] * t**2
            + 4.0 * self.coefficients[:, 4:5] * t**3
            + 5.0 * self.coefficients[:, 5:6] * t**4
        )
        qdd = (
            2.0 * self.coefficients[:, 2:3]
            + 6.0 * self.coefficients[:, 3:4] * t
            + 12.0 * self.coefficients[:, 4:5] * t**2
            + 20.0 * self.coefficients[:, 5:6] * t**3
        )
        return q, qd, qdd
```

Export it from `src/robot_calibration/trajectory/__init__.py`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_trajectory.py::test_boundary_polynomial_samples_position_velocity_and_acceleration -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_trajectory.py src/robot_calibration/trajectory/fourier.py src/robot_calibration/trajectory/__init__.py
git commit -m "feat: add boundary polynomial trajectory object"
```

---

### Task 3: Mixed Excitation Trajectory Object

**Files:**
- Modify: `tests/test_trajectory.py`
- Modify: `src/robot_calibration/trajectory/fourier.py`
- Modify: `src/robot_calibration/trajectory/__init__.py`

- [ ] **Step 1: Write the failing mixed trajectory boundary test**

Add this import:

```python
from robot_calibration.trajectory import MixedExcitationTrajectory
```

Add this test:

```python
def test_mixed_excitation_trajectory_enforces_rest_boundaries():
    duration = 2.0
    time = np.linspace(0.0, duration, 50)
    q0 = np.array([0.5])
    a = np.array([[0.2, -0.1]])
    b = np.array([[0.1, 0.05]])
    trajectory = MixedExcitationTrajectory.from_fourier_boundary(
        duration,
        q0,
        a,
        b,
        np.pi,
    )

    q, qd, qdd = trajectory.sample(time)

    np.testing.assert_allclose(q[:, 0], q0, atol=1e-12)
    np.testing.assert_allclose(q[:, -1], q0, atol=1e-12)
    np.testing.assert_allclose(qd[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(qd[:, -1], 0.0, atol=1e-12)
    np.testing.assert_allclose(qdd[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(qdd[:, -1], 0.0, atol=1e-12)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_trajectory.py::test_mixed_excitation_trajectory_enforces_rest_boundaries -v`

Expected: FAIL with an import error because `MixedExcitationTrajectory` is not exported yet.

- [ ] **Step 3: Implement the minimal mixed trajectory object**

In `src/robot_calibration/trajectory/fourier.py`, add:

```python
class MixedExcitationTrajectory:
    """Fourier excitation trajectory with fifth-order boundary correction."""

    def __init__(
        self,
        fourier: FourierExcitationTrajectory,
        boundary_polynomial: FifthOrderBoundaryPolynomial,
    ) -> None:
        if fourier.n_joints != boundary_polynomial.n_joints:
            raise ValueError("fourier and boundary polynomial joint counts must match")
        self.fourier = fourier
        self.boundary_polynomial = boundary_polynomial

    @classmethod
    def from_fourier_boundary(
        cls,
        duration: float,
        q0: ArrayLike,
        sine_coefficients: ArrayLike,
        cosine_coefficients: ArrayLike,
        fundamental_frequency: float,
    ) -> "MixedExcitationTrajectory":
        fourier = FourierExcitationTrajectory(
            np.zeros_like(np.asarray(q0, dtype=float)),
            sine_coefficients,
            cosine_coefficients,
            fundamental_frequency,
        )
        boundary = FifthOrderBoundaryPolynomial.from_fourier_boundary(
            duration,
            q0,
            sine_coefficients,
            cosine_coefficients,
            fundamental_frequency,
        )
        return cls(fourier, boundary)

    @property
    def n_joints(self) -> int:
        return self.fourier.n_joints

    def sample(
        self,
        time: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        q_fourier, qd_fourier, qdd_fourier = self.fourier.sample(time)
        q_boundary, qd_boundary, qdd_boundary = self.boundary_polynomial.sample(time)
        return (
            q_fourier + q_boundary,
            qd_fourier + qd_boundary,
            qdd_fourier + qdd_boundary,
        )
```

Export it from `src/robot_calibration/trajectory/__init__.py`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_trajectory.py::test_mixed_excitation_trajectory_enforces_rest_boundaries -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_trajectory.py src/robot_calibration/trajectory/fourier.py src/robot_calibration/trajectory/__init__.py
git commit -m "feat: add mixed excitation trajectory object"
```

---

### Task 4: Uniform Sampling And Generator

**Files:**
- Modify: `tests/test_trajectory.py`
- Modify: `src/robot_calibration/trajectory/fourier.py`
- Modify: `src/robot_calibration/trajectory/__init__.py`

- [ ] **Step 1: Write the failing uniform sampling and generator test**

Add this import:

```python
from robot_calibration.trajectory import ExcitationTrajectoryGenerator
```

Add this test:

```python
def test_excitation_generator_builds_uniformly_sampled_mixed_trajectory():
    q0 = np.array([0.1, -0.2])
    a = np.array([[0.2, -0.1], [0.05, 0.15]])
    b = np.array([[0.1, 0.05], [-0.1, 0.02]])
    generator = ExcitationTrajectoryGenerator(
        q0=q0,
        sine_coefficients=a,
        cosine_coefficients=b,
        fundamental_frequency=np.pi,
        duration=2.0,
    )

    trajectory = generator.build()
    time, q, qd, qdd = trajectory.sample_uniform(sample_count=21)

    assert time.shape == (21,)
    assert q.shape == (2, 21)
    assert qd.shape == (2, 21)
    assert qdd.shape == (2, 21)
    np.testing.assert_allclose(time[[0, -1]], [0.0, 2.0])
    np.testing.assert_allclose(q[:, 0], q0, atol=1e-12)
    np.testing.assert_allclose(q[:, -1], q0, atol=1e-12)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_trajectory.py::test_excitation_generator_builds_uniformly_sampled_mixed_trajectory -v`

Expected: FAIL with an import error because `ExcitationTrajectoryGenerator` is not exported yet.

- [ ] **Step 3: Implement uniform sampling and generator**

Add `duration` support and uniform sampling to `MixedExcitationTrajectory`:

```python
class MixedExcitationTrajectory:
    def __init__(
        self,
        fourier: FourierExcitationTrajectory,
        boundary_polynomial: FifthOrderBoundaryPolynomial,
        duration: float | None = None,
    ) -> None:
        if fourier.n_joints != boundary_polynomial.n_joints:
            raise ValueError("fourier and boundary polynomial joint counts must match")
        if duration is not None and duration <= 0.0:
            raise ValueError("duration must be positive")
        self.fourier = fourier
        self.boundary_polynomial = boundary_polynomial
        self.duration = None if duration is None else float(duration)

    def sample_uniform(
        self,
        sample_count: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if self.duration is None:
            raise ValueError("duration is required for uniform sampling")
        if sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        time = np.linspace(0.0, self.duration, sample_count)
        q, qd, qdd = self.sample(time)
        return time, q, qd, qdd
```

Add the generator:

```python
class ExcitationTrajectoryGenerator:
    """Factory for boundary-corrected Fourier excitation trajectories."""

    def __init__(
        self,
        q0: ArrayLike,
        sine_coefficients: ArrayLike,
        cosine_coefficients: ArrayLike,
        fundamental_frequency: float,
        duration: float,
    ) -> None:
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        self.q0 = np.asarray(q0, dtype=float)
        self.sine_coefficients, self.cosine_coefficients = _coefficient_arrays(
            sine_coefficients,
            cosine_coefficients,
        )
        if self.q0.shape != (self.sine_coefficients.shape[0],):
            raise ValueError("q0 must have one entry per joint")
        if fundamental_frequency <= 0.0:
            raise ValueError("fundamental_frequency must be positive")
        self.fundamental_frequency = float(fundamental_frequency)
        self.duration = float(duration)

    def build(self) -> MixedExcitationTrajectory:
        return MixedExcitationTrajectory.from_fourier_boundary(
            self.duration,
            self.q0,
            self.sine_coefficients,
            self.cosine_coefficients,
            self.fundamental_frequency,
        )
```

Update `MixedExcitationTrajectory.from_fourier_boundary()` to pass `duration=duration`.

Export `ExcitationTrajectoryGenerator`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_trajectory.py::test_excitation_generator_builds_uniformly_sampled_mixed_trajectory -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_trajectory.py src/robot_calibration/trajectory/fourier.py src/robot_calibration/trajectory/__init__.py
git commit -m "feat: add excitation trajectory generator"
```

---

### Task 5: Validation Coverage And Compatibility

**Files:**
- Modify: `tests/test_trajectory.py`
- Modify: `src/robot_calibration/trajectory/fourier.py`

- [ ] **Step 1: Write failing validation tests**

Add these tests:

```python
def test_fourier_excitation_trajectory_rejects_invalid_time_shape():
    trajectory = FourierExcitationTrajectory(
        np.zeros(1),
        np.ones((1, 2)),
        np.zeros((1, 2)),
        np.pi,
    )

    with pytest.raises(ValueError, match="time must be 1D"):
        trajectory.sample(np.zeros((2, 2)))


def test_mixed_excitation_trajectory_rejects_uniform_sampling_without_duration():
    fourier = FourierExcitationTrajectory(
        np.zeros(1),
        np.ones((1, 2)),
        np.zeros((1, 2)),
        np.pi,
    )
    boundary = FifthOrderBoundaryPolynomial(np.zeros((1, 6)))
    trajectory = MixedExcitationTrajectory(fourier, boundary)

    with pytest.raises(ValueError, match="duration is required"):
        trajectory.sample_uniform(10)
```

Ensure `tests/test_trajectory.py` imports `pytest`.

- [ ] **Step 2: Run the validation tests to verify they fail if needed**

Run:

```bash
pytest \
  tests/test_trajectory.py::test_fourier_excitation_trajectory_rejects_invalid_time_shape \
  tests/test_trajectory.py::test_mixed_excitation_trajectory_rejects_uniform_sampling_without_duration \
  -v
```

Expected: FAIL only for missing imports or missing validation behavior. If they pass because earlier code already covers the behavior, keep them as regression tests.

- [ ] **Step 3: Add or tighten validation**

Keep validation messages consistent with existing helpers:

- `time must be 1D`
- `duration must be positive`
- `sample_count must be at least 2`
- `q0 must have one entry per joint`
- `sine and cosine coefficients must be matching 2D arrays`

- [ ] **Step 4: Run trajectory tests**

Run: `pytest tests/test_trajectory.py -v`

Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run: `pytest -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_trajectory.py src/robot_calibration/trajectory/fourier.py
git commit -m "test: cover excitation trajectory validation"
```

---

## Notes

- Keep existing function helpers public and behavior-compatible.
- Avoid adding `pymoo` or optimizer objects in this plan; that belongs to the next roadmap stage.
- The class methods should return arrays in the same `(n_joints, n_samples)` layout as the existing helpers.
- The implementation should not change README, `docs/features.md`, or `docs/roadmap.md` unless the user asks for docs updates after code lands.
