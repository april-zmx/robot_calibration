import subprocess
import sys


def test_package_imports():
    import robot_calibration

    assert robot_calibration.__version__


def test_cli_help_importable():
    result = subprocess.run(
        [sys.executable, "-m", "robot_calibration.cli.main", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "robot-calib" in result.stdout
