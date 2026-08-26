"""G1 provider namespace with a cold-start-safe public contract.

Concrete policies live in their modules.  Keeping this package entry point
side-effect free prevents the simulator/world contract from importing Growth
while Growth is still importing the same world contract.
"""

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES

__all__ = [
    "G1_DDS_JOINT_NAMES",
]
