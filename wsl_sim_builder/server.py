"""WSL-side MuJoCo simulation builder service.

This is intentionally independent from the wheel-leg demo files.  A Windows
browser can POST a mechanical structure JSON and a Python control function; the
server generates MJCF, runs a headless MuJoCo simulation, and returns sampled
trajectory data for the frontend.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import uuid
import xml.dom.minidom
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
FRONTEND = ROOT / "frontend" / "index.html"
DEMO_REQUEST = ROOT / "demo_request.json"
GENERATED = ROOT / "generated"
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class UserFacingError(Exception):
    """Error that can be shown directly in the frontend."""


def ensure_name(value: str, kind: str) -> str:
    if not isinstance(value, str) or not NAME_RE.match(value):
        raise UserFacingError(
            f"{kind} name must match [A-Za-z_][A-Za-z0-9_]*, got {value!r}"
        )
    return value


def vector(value: object, expected: int | None = None) -> str:
    if not isinstance(value, list | tuple):
        raise UserFacingError(f"Expected a numeric list, got {value!r}")
    if expected is not None and len(value) != expected:
        raise UserFacingError(f"Expected {expected} values, got {len(value)}")
    return " ".join(f"{float(item):.12g}" for item in value)


def maybe_set(elem: ET.Element, source: dict, key: str, *, as_vector: int | None = None) -> None:
    if key not in source or source[key] is None:
        return
    if as_vector is None:
        elem.set(key, str(source[key]))
    else:
        elem.set(key, vector(source[key], as_vector))


def append_geom(parent: ET.Element, geom: dict) -> None:
    geom_type = geom.get("type", "box")
    if geom_type not in {"box", "sphere", "capsule", "cylinder", "plane"}:
        raise UserFacingError(f"Unsupported geom type: {geom_type}")
    attrs = {"type": geom_type}
    if "name" in geom:
        attrs["name"] = ensure_name(geom["name"], "geom")
    elem = ET.SubElement(parent, "geom", attrs)
    for key in ("pos", "size", "fromto", "rgba", "euler", "friction"):
        if key in geom:
            elem.set(key, vector(geom[key]))
    for key in ("mass", "density", "group", "contype", "conaffinity"):
        maybe_set(elem, geom, key)


def append_joint(parent: ET.Element, joint: dict) -> None:
    name = ensure_name(joint["name"], "joint")
    joint_type = joint.get("type", "hinge")
    if joint_type not in {"free", "ball", "slide", "hinge"}:
        raise UserFacingError(f"Unsupported joint type: {joint_type}")
    elem = ET.SubElement(parent, "joint", {"name": name, "type": joint_type})
    for key in ("axis", "pos", "range"):
        if key in joint:
            elem.set(key, vector(joint[key]))
    for key in ("damping", "armature", "limited"):
        maybe_set(elem, joint, key)


def build_mjcf(structure: dict) -> str:
    model_name = ensure_name(structure.get("name", "generated_model"), "model")
    timestep = float(structure.get("timestep", 0.002))
    gravity = structure.get("gravity", [0, 0, -9.81])
    if timestep <= 0.0 or timestep > 0.05:
        raise UserFacingError("timestep must be in (0, 0.05]")

    root = ET.Element("mujoco", {"model": model_name})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": f"{timestep:.12g}",
            "gravity": vector(gravity, 3),
            "solver": structure.get("solver", "Newton"),
            "iterations": str(int(structure.get("iterations", 80))),
        },
    )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 -3 4", "dir": "0 1 -1"})
    if structure.get("include_floor", True):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "size": "10 10 .1",
                "rgba": ".22 .24 .25 1",
                "friction": "1.0 0.01 0.001",
            },
        )
    for geom in structure.get("world_geoms", []):
        append_geom(worldbody, geom)

    bodies = structure.get("bodies", [])
    if not isinstance(bodies, list):
        raise UserFacingError("structure.bodies must be a list")
    body_by_name: dict[str, dict] = {}
    children: dict[str, list[dict]] = {"world": []}
    for body in bodies:
        name = ensure_name(body["name"], "body")
        if name in body_by_name:
            raise UserFacingError(f"Duplicate body name: {name}")
        body_by_name[name] = body
    for body in bodies:
        parent_name = body.get("parent", "world")
        if parent_name != "world" and parent_name not in body_by_name:
            raise UserFacingError(f"Body {body['name']} parent does not exist: {parent_name}")
        children.setdefault(parent_name, []).append(body)

    def append_body(parent_elem: ET.Element, body: dict) -> None:
        elem = ET.SubElement(parent_elem, "body", {"name": body["name"]})
        if "pos" in body:
            elem.set("pos", vector(body["pos"], 3))
        if "euler" in body:
            elem.set("euler", vector(body["euler"], 3))
        joints = body.get("joints", [])
        if isinstance(joints, dict):
            joints = [joints]
        for joint in joints:
            append_joint(elem, joint)
        for geom in body.get("geoms", []):
            append_geom(elem, geom)
        for child in children.get(body["name"], []):
            append_body(elem, child)

    for body in children.get("world", []):
        append_body(worldbody, body)

    actuators = structure.get("actuators", [])
    if actuators:
        actuator_root = ET.SubElement(root, "actuator")
        for actuator in actuators:
            name = ensure_name(actuator["name"], "actuator")
            joint = ensure_name(actuator["joint"], "actuator joint")
            elem = ET.SubElement(actuator_root, "motor", {"name": name, "joint": joint})
            for key in ("gear", "ctrllimited"):
                maybe_set(elem, actuator, key)
            if "ctrlrange" in actuator:
                elem.set("ctrlrange", vector(actuator["ctrlrange"], 2))
                elem.set("ctrllimited", "true")

    rough = ET.tostring(root, encoding="unicode")
    return xml.dom.minidom.parseString(rough).toprettyxml(indent="  ")


SAFE_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "dict": dict,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "sum": sum,
}


def compile_control(code: str):
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
        "np": np,
        "numpy": np,
    }
    try:
        exec(code, namespace, namespace)
    except Exception as exc:  # pragma: no cover - surfaced to frontend.
        raise UserFacingError(f"Control code failed to compile: {exc}") from exc
    control = namespace.get("control")
    if not callable(control):
        raise UserFacingError("Control code must define: control(t, qpos, qvel, ctrl, state)")
    return control


def joint_widths(model: mujoco.MjModel, joint_id: int) -> tuple[int, int]:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7, 6
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4, 3
    return 1, 1


def named_joint_values(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict, dict]:
    qpos: dict[str, float | list[float]] = {}
    qvel: dict[str, float | list[float]] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        qpos_width, qvel_width = joint_widths(model, joint_id)
        qpos_addr = int(model.jnt_qposadr[joint_id])
        qvel_addr = int(model.jnt_dofadr[joint_id])
        qpos_slice = data.qpos[qpos_addr : qpos_addr + qpos_width]
        qvel_slice = data.qvel[qvel_addr : qvel_addr + qvel_width]
        qpos[name] = float(qpos_slice[0]) if qpos_width == 1 else qpos_slice.tolist()
        qvel[name] = float(qvel_slice[0]) if qvel_width == 1 else qvel_slice.tolist()
    return qpos, qvel


def actuator_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
        or f"actuator_{idx}"
        for idx in range(model.nu)
    ]


def body_positions(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, list[float]]:
    result = {}
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name:
            result[name] = [float(x) for x in data.xpos[body_id]]
    return result


GEOM_TYPE_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
}


def geom_display_name(model: mujoco.MjModel, geom_id: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        or f"geom_{geom_id}"
    )


def geom_descriptors(model: mujoco.MjModel) -> list[dict]:
    result = []
    for geom_id in range(model.ngeom):
        result.append(
            {
                "id": geom_id,
                "name": geom_display_name(model, geom_id),
                "type": GEOM_TYPE_NAMES.get(int(model.geom_type[geom_id]), "other"),
                "size": [float(x) for x in model.geom_size[geom_id]],
                "rgba": [float(x) for x in model.geom_rgba[geom_id]],
            }
        )
    return result


def geom_poses(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    return [
        {
            "name": geom_display_name(model, geom_id),
            "pos": [float(x) for x in data.geom_xpos[geom_id]],
            "xmat": [float(x) for x in data.geom_xmat[geom_id]],
        }
        for geom_id in range(model.ngeom)
    ]


def run_generated_simulation(xml: str, code: str, duration: float, sample_hz: float) -> dict:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    control = compile_control(code)
    names = actuator_names(model)
    actuator_id = {name: idx for idx, name in enumerate(names)}
    duration = max(0.01, min(float(duration), 30.0))
    sample_hz = max(1.0, min(float(sample_hz), 240.0))
    dt = float(model.opt.timestep)
    steps = min(int(duration / dt), 200000)
    sample_interval = max(1, int(round(1.0 / (sample_hz * dt))))
    series = []
    max_abs_ctrl = {name: 0.0 for name in names}
    visual_model = {"geoms": geom_descriptors(model)}

    for step in range(steps + 1):
        qpos, qvel = named_joint_values(model, data)
        ctrl = {name: 0.0 for name in names}
        state = {
            "step": step,
            "body_pos": body_positions(model, data),
            "actuators": names,
            "qpos_array": data.qpos.tolist(),
            "qvel_array": data.qvel.tolist(),
        }
        try:
            returned = control(float(data.time), qpos, qvel, ctrl, state)
        except Exception as exc:
            raise UserFacingError(
                "Control code raised an error at "
                f"t={data.time:.4f}s: {exc}\n{traceback.format_exc(limit=4)}"
            ) from exc
        if isinstance(returned, dict):
            ctrl.update(returned)
        for name, value in ctrl.items():
            if name not in actuator_id:
                raise UserFacingError(f"Control set unknown actuator {name!r}")
            aid = actuator_id[name]
            numeric = float(value)
            if model.actuator_ctrllimited[aid]:
                lo, hi = model.actuator_ctrlrange[aid]
                numeric = float(np.clip(numeric, lo, hi))
            data.ctrl[aid] = numeric
            max_abs_ctrl[name] = max(max_abs_ctrl[name], abs(numeric))

        if step % sample_interval == 0 or step == steps:
            series.append(
                {
                    "t": float(data.time),
                    "qpos": qpos,
                    "qvel": qvel,
                    "ctrl": {name: float(data.ctrl[idx]) for name, idx in actuator_id.items()},
                    "body_pos": body_positions(model, data),
                    "geom_pose": geom_poses(model, data),
                }
            )
        if step < steps:
            mujoco.mj_step(model, data)

    final_qpos, final_qvel = named_joint_values(model, data)
    return {
        "duration": duration,
        "timestep": dt,
        "steps": steps,
        "samples": len(series),
        "joints": list(final_qpos.keys()),
        "actuators": names,
        "visual_model": visual_model,
        "series": series,
        "summary": {
            "final_time": float(data.time),
            "final_qpos": final_qpos,
            "final_qvel": final_qvel,
            "max_abs_ctrl": max_abs_ctrl,
        },
    }


def save_artifacts(request_payload: dict, xml: str, result: dict | None = None) -> str:
    GENERATED.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    folder = GENERATED / run_id
    folder.mkdir()
    (folder / "scene.xml").write_text(xml, encoding="utf-8")
    (folder / "request.json").write_text(
        json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result is not None:
        light_result = dict(result)
        # Keep browser payload complete, but artifact compact enough to inspect.
        (folder / "result.json").write_text(
            json.dumps(light_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return run_id


def launch_process(command: list[str], run_id: str, log_name: str) -> dict:
    folder = GENERATED / run_id
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / log_name
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {
        "pid": process.pid,
        "log": f"/api/files/{run_id}/{log_name}",
    }


def load_demo_payload() -> dict:
    return json.loads(DEMO_REQUEST.read_text(encoding="utf-8"))


def generate(payload: dict) -> dict:
    structure = payload.get("structure")
    if not isinstance(structure, dict):
        raise UserFacingError("Request must contain a structure object")
    xml = build_mjcf(structure)
    run_id = save_artifacts(payload, xml)
    return {
        "ok": True,
        "run_id": run_id,
        "xml": xml,
        "files": {
            "scene_xml": f"/api/files/{run_id}/scene.xml",
            "request_json": f"/api/files/{run_id}/request.json",
        },
    }


def run(payload: dict) -> dict:
    structure = payload.get("structure")
    code = payload.get("control_code", "")
    if not isinstance(structure, dict):
        raise UserFacingError("Request must contain a structure object")
    if not isinstance(code, str) or not code.strip():
        raise UserFacingError("Request must contain non-empty control_code")
    xml = build_mjcf(structure)
    result = run_generated_simulation(
        xml,
        code,
        payload.get("duration", structure.get("duration", 5.0)),
        payload.get("sample_hz", 60.0),
    )
    run_id = save_artifacts(payload, xml, result)
    result.update(
        {
            "ok": True,
            "run_id": run_id,
            "xml": xml,
            "files": {
                "scene_xml": f"/api/files/{run_id}/scene.xml",
                "request_json": f"/api/files/{run_id}/request.json",
                "result_json": f"/api/files/{run_id}/result.json",
            },
        }
    )
    return result


def launch_wheel_leg(payload: dict) -> dict:
    run_id = "wheel-leg-" + time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    folder = GENERATED / run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "request.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    command = [sys.executable, str(PROJECT_ROOT / "Simulation.py")]
    if payload.get("detailed_meshes"):
        command.append("--detailed-meshes")
    if payload.get("hide_control_hud"):
        command.append("--hide-control-hud")
    if payload.get("status_interval") is not None:
        command.extend(["--status-interval", str(float(payload["status_interval"]))])
    launched = launch_process(command, run_id, "wheel_leg_gui.log")
    return {
        "ok": True,
        "mode": "wheel_leg_gui",
        "run_id": run_id,
        "pid": launched["pid"],
        "message": "已在 WSL 中启动轮腿完整 MuJoCo 图形仿真。",
        "files": {
            "request_json": f"/api/files/{run_id}/request.json",
            "gui_log": launched["log"],
        },
    }


def launch_generated_gui(payload: dict) -> dict:
    structure = payload.get("structure")
    code = payload.get("control_code", "")
    if not isinstance(structure, dict):
        raise UserFacingError("Request must contain a structure object")
    if not isinstance(code, str) or not code.strip():
        raise UserFacingError("Request must contain non-empty control_code")
    xml = build_mjcf(structure)
    run_id = save_artifacts(payload, xml)
    request_path = GENERATED / run_id / "request.json"
    scene_path = GENERATED / run_id / "scene.xml"
    command = [
        sys.executable,
        str(ROOT / "gui_runner.py"),
        "--request",
        str(request_path),
        "--xml",
        str(scene_path),
    ]
    launched = launch_process(command, run_id, "generated_gui.log")
    return {
        "ok": True,
        "mode": "generated_gui",
        "run_id": run_id,
        "pid": launched["pid"],
        "xml": xml,
        "message": "已在 WSL 中启动当前生成模型的完整 MuJoCo 图形仿真。",
        "files": {
            "scene_xml": f"/api/files/{run_id}/scene.xml",
            "request_json": f"/api/files/{run_id}/request.json",
            "gui_log": launched["log"],
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "WSLSimBuilder/0.1"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.end_headers()

    def send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_json(404, {"ok": False, "error": "file not found"})
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise UserFacingError("Missing request body")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            self.send_text_file(FRONTEND, "text/html; charset=utf-8")
            return
        if path == "/api/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": "wsl_sim_builder",
                    "features": ["headless_run", "generated_gui", "wheel_leg_gui"],
                },
            )
            return
        if path == "/api/demo":
            self.send_json(200, {"ok": True, "payload": load_demo_payload()})
            return
        if path.startswith("/api/files/"):
            parts = path.strip("/").split("/")
            if len(parts) != 4:
                self.send_json(404, {"ok": False, "error": "bad file path"})
                return
            _, _, run_id, filename = parts
            if not RUN_ID_RE.match(run_id):
                self.send_json(400, {"ok": False, "error": "bad run id"})
                return
            if filename not in {
                "scene.xml",
                "request.json",
                "result.json",
                "generated_gui.log",
                "wheel_leg_gui.log",
            }:
                self.send_json(404, {"ok": False, "error": "unsupported file"})
                return
            if filename.endswith(".xml"):
                content_type = "application/xml"
            elif filename.endswith(".log"):
                content_type = "text/plain; charset=utf-8"
            else:
                content_type = "application/json"
            self.send_text_file(GENERATED / run_id / filename, content_type)
            return
        self.send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            payload = self.read_json()
            if path == "/api/generate":
                self.send_json(200, generate(payload))
                return
            if path == "/api/run":
                self.send_json(200, run(payload))
                return
            if path == "/api/launch/wheel-leg":
                self.send_json(200, launch_wheel_leg(payload))
                return
            if path == "/api/launch/generated":
                self.send_json(200, launch_generated_gui(payload))
                return
            self.send_json(404, {"ok": False, "error": "not found"})
        except UserFacingError as exc:
            self.send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive server boundary.
            self.send_json(
                500,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                },
            )

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def self_test() -> None:
    payload = load_demo_payload()
    result = run(payload)
    print(
        "self-test ok:",
        f"run_id={result['run_id']}",
        f"steps={result['steps']}",
        f"samples={result['samples']}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    GENERATED.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"WSL Sim Builder running at http://{args.host}:{args.port}")
    print("Open http://localhost:%d in Windows Chrome/Edge." % args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
