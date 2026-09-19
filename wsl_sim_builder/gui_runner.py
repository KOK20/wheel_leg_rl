"""Launch a full MuJoCo GUI viewer for a generated model request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from server import (
    actuator_names,
    body_positions,
    build_mjcf,
    compile_control,
    named_joint_values,
)


def apply_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control,
    actuator_id: dict[str, int],
) -> None:
    names = list(actuator_id)
    qpos, qvel = named_joint_values(model, data)
    ctrl = {name: 0.0 for name in names}
    state = {
        "body_pos": body_positions(model, data),
        "actuators": names,
        "qpos_array": data.qpos.tolist(),
        "qvel_array": data.qvel.tolist(),
    }
    returned = control(float(data.time), qpos, qvel, ctrl, state)
    if isinstance(returned, dict):
        ctrl.update(returned)
    for name, value in ctrl.items():
        if name not in actuator_id:
            raise RuntimeError(f"Control set unknown actuator {name!r}")
        aid = actuator_id[name]
        numeric = float(value)
        if model.actuator_ctrllimited[aid]:
            lo, hi = model.actuator_ctrlrange[aid]
            numeric = float(np.clip(numeric, lo, hi))
        data.ctrl[aid] = numeric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--xml", required=True)
    args = parser.parse_args()

    request_path = Path(args.request)
    xml_path = Path(args.xml)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    xml = (
        xml_path.read_text(encoding="utf-8")
        if xml_path.exists()
        else build_mjcf(payload["structure"])
    )
    control = compile_control(payload.get("control_code", ""))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = actuator_names(model)
    actuator_id = {name: idx for idx, name in enumerate(names)}
    dt = float(model.opt.timestep)

    print("Generated MuJoCo GUI started.")
    print(f"request={request_path}")
    print(f"xml={xml_path}")
    print(f"actuators={names}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            apply_control(model, data, control, actuator_id)
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
