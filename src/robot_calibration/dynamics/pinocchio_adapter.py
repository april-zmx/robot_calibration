"""Optional Pinocchio-backed inverse-dynamics regressors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _load_pinocchio() -> Any:
    try:
        import pinocchio as pin
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pinocchio is required for PinocchioRegressor; install the optional robotics dependencies"
        ) from exc
    return pin


def _load_robot_wrapper() -> Any:
    try:
        from pinocchio.robot_wrapper import RobotWrapper
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pinocchio.robot_wrapper is required for URDF loading"
        ) from exc
    return RobotWrapper


@dataclass(frozen=True)
class PinocchioRegressor:
    """Callable wrapper around ``pin.computeJointTorqueRegressor``.

    This mirrors the notebook workflow:
    ``Y = pin.computeJointTorqueRegressor(model, data, q, qd, qdd)``.
    When motor dynamics are enabled, one reflected-inertia column ``qdd_i``
    is inserted after each link's dynamic parameter block.
    """

    robot: Any
    include_motor_dynamics: bool = False
    link_parameter_count: int = 10
    pinocchio: Any | None = "auto"

    def __post_init__(self) -> None:
        if self.link_parameter_count <= 0:
            raise ValueError("link_parameter_count must be positive")
        if self.pinocchio == "auto":
            object.__setattr__(self, "pinocchio", _load_pinocchio())
        elif self.pinocchio is None:
            raise ModuleNotFoundError(
                "pinocchio is required for PinocchioRegressor; install the optional robotics dependencies"
            )

    @property
    def nq(self) -> int:
        return int(self.robot.model.nq)

    def __call__(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        qdd: ArrayLike,
    ) -> NDArray[np.float64]:
        q_array = np.asarray(q, dtype=float)
        qd_array = np.asarray(qd, dtype=float)
        qdd_array = np.asarray(qdd, dtype=float)
        expected = (self.nq,)
        if q_array.shape != expected or qd_array.shape != expected or qdd_array.shape != expected:
            raise ValueError(f"q, qd, and qdd must all have shape {expected}")

        y = np.asarray(
            self.pinocchio.computeJointTorqueRegressor(
                self.robot.model,
                self.robot.data,
                q_array,
                qd_array,
                qdd_array,
            ),
            dtype=float,
        )
        if not self.include_motor_dynamics:
            return y
        return self._with_motor_dynamics(y, qdd_array)

    def _with_motor_dynamics(
        self,
        regressor: NDArray[np.float64],
        qdd: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        expected_columns = self.nq * self.link_parameter_count
        if regressor.shape != (self.nq, expected_columns):
            raise ValueError(
                "Pinocchio regressor shape is incompatible with nq and link_parameter_count"
            )

        blocks = []
        motor_columns = np.diag(qdd)
        for joint in range(self.nq):
            start = joint * self.link_parameter_count
            stop = start + self.link_parameter_count
            blocks.append(regressor[:, start:stop])
            blocks.append(motor_columns[:, joint : joint + 1])
        return np.concatenate(blocks, axis=1)


def build_pinocchio_regressor_from_urdf(
    urdf_path: str | Path,
    *,
    include_motor_dynamics: bool = False,
    link_parameter_count: int = 10,
    verbose: bool = False,
    pinocchio: Any | None = "auto",
    robot_wrapper_cls: Any | None = "auto",
) -> PinocchioRegressor:
    """Build a :class:`PinocchioRegressor` from a URDF path."""

    pin = _load_pinocchio() if pinocchio == "auto" else pinocchio
    if pin is None:
        raise ModuleNotFoundError(
            "pinocchio is required for build_pinocchio_regressor_from_urdf"
        )
    wrapper_cls = _load_robot_wrapper() if robot_wrapper_cls == "auto" else robot_wrapper_cls
    if wrapper_cls is None:
        raise ModuleNotFoundError(
            "pinocchio.robot_wrapper is required for build_pinocchio_regressor_from_urdf"
        )

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_path),
        verbose=verbose,
    )
    robot = wrapper_cls(model, collision_model, visual_model)
    return PinocchioRegressor(
        robot,
        include_motor_dynamics=include_motor_dynamics,
        link_parameter_count=link_parameter_count,
        pinocchio=pin,
    )


def build_pinocchio_model_regressor_from_urdf(
    urdf_path: str | Path,
    *,
    include_motor_dynamics: bool = False,
    link_parameter_count: int = 10,
    pinocchio: Any | None = "auto",
) -> PinocchioRegressor:
    """Build a regressor from a URDF model without loading geometry meshes."""

    pin = _load_pinocchio() if pinocchio == "auto" else pinocchio
    if pin is None:
        raise ModuleNotFoundError(
            "pinocchio is required for build_pinocchio_model_regressor_from_urdf"
        )
    model = pin.buildModelFromUrdf(str(urdf_path))
    robot = SimpleNamespace(model=model, data=model.createData())
    return PinocchioRegressor(
        robot,
        include_motor_dynamics=include_motor_dynamics,
        link_parameter_count=link_parameter_count,
        pinocchio=pin,
    )
