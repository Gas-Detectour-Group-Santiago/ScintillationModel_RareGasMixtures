from __future__ import annotations

from ArCF4_fit import CONFIG as ARCF4_CONFIG
from ArCF4_IR_fit import CONFIG as ARCF4_IR_CONFIG
from ArN2_fit import CONFIG as ARN2_CONFIG
from ArN2_IR_fit import CONFIG as ARN2_IR_CONFIG
from auxiliares import PrimaryFitRunner


CONFIGS = [
    ARCF4_CONFIG,
    ARN2_CONFIG,
    ARCF4_IR_CONFIG,
    ARN2_IR_CONFIG,
]


if __name__ == "__main__":
    for config in CONFIGS:
        PrimaryFitRunner(config).run_all()
