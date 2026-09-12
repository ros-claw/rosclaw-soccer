"""Inspect the current OpenGL renderer without opening a robot or changing drivers.

Driver-reported identity is not a performance benchmark or physics certificate.
In particular, Mesa is not synonymous with software: inspect the renderer too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RendererBackendIdentity:
    vendor: str
    renderer: str
    version: str
    reported_device_kind: Literal["software", "nvidia_gpu", "unknown"]
    nvidia_driver_version: str | None


def parse_renderer_identity(
    vendor: bytes | None, renderer: bytes | None, version: bytes | None
) -> RendererBackendIdentity:
    """Decode bounded GL strings; an absent current context is an error."""
    values = []
    for raw in (vendor, renderer, version):
        if type(raw) is not bytes or not raw or len(raw) > 512:
            raise ValueError("bounded OpenGL identity bytes from a current context required")
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("OpenGL identity must be UTF-8") from error
        if not value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("OpenGL identity contains empty or control text")
        values.append(value)
    vendor_text, renderer_text, version_text = values
    lower = renderer_text.lower()
    kind: Literal["software", "nvidia_gpu", "unknown"] = "unknown"
    driver_version = None
    if any(word in lower for word in ("llvmpipe", "softpipe", "swrast", "software rasterizer")):
        kind = "software"
    elif vendor_text == "NVIDIA Corporation" and renderer_text.startswith("NVIDIA "):
        match = re.search(r"\bNVIDIA ([0-9]+\.[0-9]+\.[0-9]+)\b", version_text)
        if match:
            kind = "nvidia_gpu"
            driver_version = match.group(1)
    return RendererBackendIdentity(vendor_text, renderer_text, version_text, kind, driver_version)


def inspect_current_renderer() -> RendererBackendIdentity:
    """Query an already-current rendering context, never create/reconfigure one."""
    from OpenGL import GL  # type: ignore[import-untyped]

    return parse_renderer_identity(
        GL.glGetString(GL.GL_VENDOR),
        GL.glGetString(GL.GL_RENDERER),
        GL.glGetString(GL.GL_VERSION),
    )


def require_nvidia_renderer(
    identity: RendererBackendIdentity, *, expected_driver_version: str
) -> None:
    """Fail closed for an explicitly NVIDIA-qualified media workflow.

    Unknown vendors are not declared broken or software; they simply cannot
    satisfy this vendor-specific contract. No system remediation is performed.
    """
    if (
        not isinstance(identity, RendererBackendIdentity)
        or any(
            type(value) is not str
            for value in (identity.vendor, identity.renderer, identity.version)
        )
        or not isinstance(expected_driver_version, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", expected_driver_version) is None
    ):
        raise ValueError("renderer identity and explicit NVIDIA driver version required")
    # Re-derive classification rather than trust caller-constructed fields.
    observed = parse_renderer_identity(
        identity.vendor.encode(), identity.renderer.encode(), identity.version.encode()
    )
    if (
        observed != identity
        or observed.reported_device_kind != "nvidia_gpu"
        or observed.nvidia_driver_version != expected_driver_version
    ):
        raise ValueError("current renderer does not satisfy the NVIDIA driver contract")
