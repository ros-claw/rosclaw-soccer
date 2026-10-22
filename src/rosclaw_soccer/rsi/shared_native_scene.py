"""Duplicate a declared G1 subtree without altering its dynamics or assets.

Only names and references to each robot are prefixed. The floor, free ball,
materials, meshes and noncolliding goal marker remain single shared objects.
The caller must declare initial player poses before any physics; this builder
does not invent spawn positions or a football task.
"""

import copy
import re
import xml.etree.ElementTree as ET


def shared_native_scene(xml: str, *, prefixes: tuple[str, ...]) -> str:
    if (
        not 1 <= len(prefixes) <= 8
        or len(set(prefixes)) != len(prefixes)
        or any(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}/", p) is None for p in prefixes)
    ):
        raise ValueError("one-to-eight unique bounded player prefixes required")
    root = ET.fromstring(xml)
    allowed = {
        "compiler",
        "option",
        "size",
        "statistic",
        "default",
        "asset",
        "worldbody",
        "contact",
        "actuator",
        "sensor",
        "visual",
    }
    if root.tag != "mujoco" or any(e.tag not in allowed for e in root):
        raise ValueError("expanded simple declared scene without hidden sections required")
    robots = [
        (w, b)
        for w in root.findall("worldbody")
        for b in w.findall("body")
        if b.get("name") == "pelvis"
    ]
    if len(robots) != 1:
        raise ValueError("exactly one source pelvis subtree required")
    world, robot = robots[0]
    objects = {e.get("name") for e in robot.iter() if e.get("name") is not None}
    if not robot.findall("freejoint") and not robot.findall("joint[@type='free']"):
        raise ValueError("free original robot root required")

    def renamed(source: ET.Element, prefix: str) -> ET.Element:
        result = copy.deepcopy(source)
        for e in result.iter():
            if "name" in e.attrib:
                e.set("name", prefix + e.attrib["name"])
            for key in ("joint", "body1", "body2", "geom1", "geom2", "site", "objname", "target"):
                if e.get(key) in objects:
                    e.set(key, prefix + e.attrib[key])
        return result

    world.remove(robot)
    for p in prefixes:
        world.append(renamed(robot, p))
    for tag in ("actuator", "sensor", "contact"):
        for section in root.findall(tag):
            for e in list(section):
                refs = [
                    e.get(k)
                    for k in ("joint", "body1", "body2", "geom1", "geom2", "site", "objname")
                ]
                if any(x in objects for x in refs):
                    section.remove(e)
                    for p in prefixes:
                        section.append(renamed(e, p))
                elif tag in ("actuator", "sensor"):
                    raise ValueError("only original robot actuators and sensors are admitted")
    return ET.tostring(root, encoding="unicode")
