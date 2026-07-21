from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ArCF4_fit import CONFIG as ARCF4_CONFIG
from ArCF4_IR_fit import CONFIG as ARCF4_IR_CONFIG
from ArJoint_IR_fit import main as run_joint_ir_fit
from ArN2_fit import CONFIG as ARN2_CONFIG
from ArN2_IR_fit import CONFIG as ARN2_IR_CONFIG
from auxiliares import PrimaryFitRunner


CONFIGS = (
    ARCF4_CONFIG,
    ARN2_CONFIG,
    ARCF4_IR_CONFIG,
    ARN2_IR_CONFIG,
)


def main() -> None:
    for config in CONFIGS:
        PrimaryFitRunner(config).run_all()

    # Additional fit with common Ar parameters and independent CF4/N2 quenching.
    # It is run here so the project-level runner really refreshes every fit.
    run_joint_ir_fit()

    # Consolidate every freshly generated fit with the literature/kinetic
    # parameters. Predictions can then resolve parameters by mixture/model
    # instead of relying on unrelated CSV paths.
    from scintillation.parameters import build_project_parameter_registry

    build_project_parameter_registry(PROJECT_ROOT)


if __name__ == "__main__":
    main()
