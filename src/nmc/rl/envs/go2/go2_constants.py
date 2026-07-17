"""Unitree Go2 constants for the Playground joystick task.

Direct port of mujoco_playground._src.locomotion.go1.go1_constants, pointed at
our own go2_playground.xml (Menagerie's go2_mjx.xml + the sensors the Go1
joystick env needs that Menagerie doesn't ship: local_linvel, upvector, and
per-foot global_linvel). See go2/xmls/go2_playground.xml.

Go2 body/site naming differs from Go1's: root body is "base" (not "trunk"),
and foot sites are "{FR,FL,RR,RL}_foot" (not bare "FR"/"FL"/...). Foot collision
geoms keep Go1's bare naming ("FR", "FL", ...). FEET_SITES and FEET_GEOMS must
stay in the same per-leg order -- joystick.py zips them elementwise.
"""

from pathlib import Path

from mujoco_playground._src import mjx_env

ROOT_PATH = Path(__file__).resolve().parent
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_go2_playground_flat.xml"


def task_to_xml(task_name: str) -> Path:
  return {
      "flat_terrain": FLAT_TERRAIN_XML,
  }[task_name]


FEET_SITES = [
    "FR_foot",
    "FL_foot",
    "RR_foot",
    "RL_foot",
]

FEET_GEOMS = [
    "FR",
    "FL",
    "RR",
    "RL",
]

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "base"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
