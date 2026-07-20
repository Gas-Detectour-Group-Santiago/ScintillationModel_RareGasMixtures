from __future__ import annotations

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


if __name__ == "__main__":
    main()
