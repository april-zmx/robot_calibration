import numpy as np
import pytest

from robot_calibration.dynamics.base_parameters import BaseParameterMapping
from robot_calibration.trajectory import ExcitationConditionNumberProblem
from robot_calibration.trajectory import ConditionNumberExcitationOptimizer
from robot_calibration.trajectory import MixedExcitationTrajectory


def diagonal_regressor(q, qd, qdd):
    return np.array(
        [
            [q[0], qd[0], qdd[0], 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, q[1], qd[1], qdd[1]],
        ]
    )


def test_condition_number_problem_has_expected_variable_count():
    problem = ExcitationConditionNumberProblem(
        q0=np.zeros(2),
        n_harmonics=2,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=(-0.5, 0.5),
        regressor=diagonal_regressor,
    )

    assert problem.n_var == 8


def test_condition_number_problem_allows_unbounded_coefficients():
    problem = ExcitationConditionNumberProblem(
        q0=np.zeros(2),
        n_harmonics=2,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=None,
        regressor=diagonal_regressor,
    )

    assert problem.n_var == 8
    assert problem.xl is None
    assert problem.xu is None


def test_optimizer_default_initial_guess_is_seeded_random_within_bounds():
    optimizer = ConditionNumberExcitationOptimizer(
        q0=np.zeros(2),
        n_harmonics=2,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=(-0.5, 0.5),
        regressor=diagonal_regressor,
    )

    guess_a = optimizer._default_initial_guess(seed=1)
    guess_b = optimizer._default_initial_guess(seed=1)

    assert guess_a.shape == (optimizer.problem.n_var,)
    np.testing.assert_allclose(guess_a, guess_b)
    assert not np.allclose(guess_a, 0.0)
    assert np.all(guess_a >= optimizer.problem.xl)
    assert np.all(guess_a <= optimizer.problem.xu)


def test_optimizer_default_initial_guess_matches_reference_unbounded_seed_behavior():
    optimizer = ConditionNumberExcitationOptimizer(
        q0=np.zeros(2),
        n_harmonics=2,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=None,
        regressor=diagonal_regressor,
    )

    guess_a = optimizer._default_initial_guess(seed=1)
    guess_b = optimizer._default_initial_guess(seed=1)

    assert guess_a.shape == (optimizer.problem.n_var,)
    np.testing.assert_allclose(guess_a, guess_b)
    assert np.all(guess_a >= 0.0)
    assert np.all(guess_a <= 1.0)
    assert not np.allclose(guess_a, 0.0)


def test_condition_number_problem_evaluates_candidate_population():
    problem = ExcitationConditionNumberProblem(
        q0=np.array([0.1, -0.2]),
        n_harmonics=1,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=(-0.5, 0.5),
        regressor=diagonal_regressor,
    )
    out = {}

    problem._evaluate(np.zeros((2, problem.n_var)), out)

    assert out["F"].shape == (2, 1)
    assert np.all(out["F"] >= 0.0)


def test_condition_number_optimizer_returns_trajectory_result():
    pytest.importorskip("pymoo")

    optimizer = ConditionNumberExcitationOptimizer(
        q0=np.array([0.1, -0.2]),
        n_harmonics=1,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=(-0.3, 0.3),
        regressor=diagonal_regressor,
    )

    result = optimizer.optimize(max_evaluations=5, seed=1)
    time, q, qd, qdd = result.trajectory.sample_uniform(11)

    assert isinstance(result.trajectory, MixedExcitationTrajectory)
    assert result.sine_coefficients.shape == (2, 1)
    assert result.cosine_coefficients.shape == (2, 1)
    assert np.isfinite(result.condition_number)
    assert time.shape == (11,)
    assert q.shape == (2, 11)
    assert qd.shape == (2, 11)
    assert qdd.shape == (2, 11)


def test_condition_number_optimizer_starts_from_feasible_zero_coefficients():
    pytest.importorskip("pymoo")

    optimizer = ConditionNumberExcitationOptimizer(
        q0=np.zeros(2),
        n_harmonics=2,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=(-0.3, 0.3),
        regressor=diagonal_regressor,
        joint_position_limits=np.array([[-0.1, 0.1], [-0.1, 0.1]]),
        joint_velocity_limits=np.array([0.2, 0.2]),
        joint_acceleration_limits=np.array([0.3, 0.3]),
    )

    result = optimizer.optimize(max_evaluations=1, seed=1)

    assert result.decision_vector.shape == (optimizer.problem.n_var,)
    np.testing.assert_allclose(result.decision_vector, 0.0)
    assert result.used_initial_guess


def test_optimizer_prefers_feasible_zero_initial_guess_for_constrained_unbounded_problem():
    optimizer = ConditionNumberExcitationOptimizer(
        q0=np.zeros(2),
        n_harmonics=2,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=None,
        regressor=diagonal_regressor,
        joint_position_limits=np.array([[-0.1, 0.1], [-0.1, 0.1]]),
        joint_velocity_limits=np.array([0.2, 0.2]),
        joint_acceleration_limits=np.array([0.3, 0.3]),
    )

    guess = optimizer._default_initial_guess(seed=1)

    np.testing.assert_allclose(guess, 0.0)


def test_condition_number_problem_rejects_invalid_bounds_shape():
    with pytest.raises(ValueError, match="coefficient_bounds"):
        ExcitationConditionNumberProblem(
            q0=np.zeros(2),
            n_harmonics=1,
            fundamental_frequency=np.pi,
            duration=2.0,
            sample_count=11,
            coefficient_bounds=np.zeros((3, 2)),
            regressor=diagonal_regressor,
        )


def test_condition_number_problem_rejects_invalid_sample_count():
    with pytest.raises(ValueError, match="sample_count"):
        ExcitationConditionNumberProblem(
            q0=np.zeros(2),
            n_harmonics=1,
            fundamental_frequency=np.pi,
            duration=2.0,
            sample_count=1,
            coefficient_bounds=(-0.5, 0.5),
            regressor=diagonal_regressor,
        )


def test_condition_number_problem_rejects_regressor_with_wrong_joint_rows():
    def bad_regressor(q, qd, qdd):
        return np.ones((1, 3))

    problem = ExcitationConditionNumberProblem(
        q0=np.zeros(2),
        n_harmonics=1,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=(-0.5, 0.5),
        regressor=bad_regressor,
    )

    with pytest.raises(ValueError, match="one row per joint"):
        problem.evaluate_condition_number(np.zeros(problem.n_var))


def test_condition_number_problem_reports_limit_constraints():
    problem = ExcitationConditionNumberProblem(
        q0=np.zeros(2),
        n_harmonics=1,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=(-0.5, 0.5),
        regressor=diagonal_regressor,
        joint_position_limits=np.array([[-0.1, 0.1], [-0.1, 0.1]]),
        joint_velocity_limits=np.array([0.2, 0.2]),
        joint_acceleration_limits=np.array([0.3, 0.3]),
    )
    out = {}

    problem._evaluate(np.full((1, problem.n_var), 0.5), out)

    assert problem.n_ieq_constr == 12
    assert out["G"].shape == (1, 12)
    assert np.any(out["G"] > 0.0)


def test_condition_number_problem_uses_full_condition_number_for_rank_deficient_matrix():
    def rank_deficient_regressor(q, qd, qdd):
        return np.array([[q[0], qd[0], q[0]]])

    problem = ExcitationConditionNumberProblem(
        q0=np.zeros(1),
        n_harmonics=1,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=(-0.5, 0.5),
        regressor=rank_deficient_regressor,
    )

    metrics = problem.evaluate_observation_metrics(np.array([0.2, 0.1]))
    matrix = problem.observation_matrix_from_vector(np.array([0.2, 0.1]))

    assert metrics.observation_rank == 2
    assert metrics.target_rank == 3
    assert metrics.condition_number == pytest.approx(np.linalg.cond(matrix))
    assert metrics.objective_value == pytest.approx(metrics.condition_number + 1.0)
    assert metrics.condition_number > 1.0e12


def test_condition_number_problem_projects_base_regressor_and_appends_friction():
    base_mapping = BaseParameterMapping(
        rank=1,
        permutation=np.array([[0.0, 1.0], [1.0, 0.0]]),
        beta=np.zeros((1, 1)),
        pivot_indices=np.array([1, 0]),
    )

    def projected_regressor(q, qd, qdd):
        return np.array([[q[0], qd[0]]])

    def unit_friction_regressor(qd):
        return np.array([[1.0]])

    problem = ExcitationConditionNumberProblem(
        q0=np.zeros(1),
        n_harmonics=1,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=5,
        coefficient_bounds=(-0.5, 0.5),
        regressor=projected_regressor,
        base_parameter_mapping=base_mapping,
        friction_regressor=unit_friction_regressor,
    )

    decision_vector = np.array([0.3, 0.0])
    matrix = problem.observation_matrix_from_vector(decision_vector)
    _, _, qd, _ = problem.trajectory_from_vector(decision_vector).sample_uniform(5)
    expected = np.vstack(
        [np.array([[sample_qd[0], 1.0]]) for sample_qd in qd.T]
    )

    np.testing.assert_allclose(matrix, expected)


def test_condition_number_problem_penalizes_lower_rank_trajectories():
    def velocity_regressor(q, qd, qdd):
        return np.array([[1.0, qd[0]]])

    problem = ExcitationConditionNumberProblem(
        q0=np.zeros(1),
        n_harmonics=1,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=(-0.5, 0.5),
        regressor=velocity_regressor,
    )

    zero_value = problem.evaluate_objective(np.zeros(problem.n_var))
    moving_value = problem.evaluate_objective(np.array([0.5, 0.0]))

    assert moving_value < zero_value


def test_condition_number_problem_reports_condition_rank_and_objective_separately():
    def velocity_regressor(q, qd, qdd):
        return np.array([[1.0, qd[0]]])

    problem = ExcitationConditionNumberProblem(
        q0=np.zeros(1),
        n_harmonics=1,
        fundamental_frequency=np.pi,
        duration=2.0,
        sample_count=11,
        coefficient_bounds=(-0.5, 0.5),
        regressor=velocity_regressor,
        singular_penalty=123.0,
    )

    metrics = problem.evaluate_observation_metrics(np.zeros(problem.n_var))

    assert metrics.condition_number == np.inf
    assert metrics.observation_rank == 1
    assert metrics.target_rank == 2
    assert metrics.objective_value == pytest.approx(124.0)
