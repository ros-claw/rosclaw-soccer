import sys
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from rosclaw_soccer.diagnostics.renderer_backend import (
    inspect_current_renderer,
    parse_renderer_identity,
    require_nvidia_renderer,
)


def nvidia():
    return parse_renderer_identity(
        b"NVIDIA Corporation", b"NVIDIA RTX A6000/PCIe/SSE2", b"4.6.0 NVIDIA 595.71.05"
    )


def test_nvidia_identity_requires_explicit_matching_driver():
    identity = nvidia()
    assert identity.reported_device_kind == "nvidia_gpu"
    require_nvidia_renderer(identity, expected_driver_version="595.71.05")
    with pytest.raises(ValueError):
        require_nvidia_renderer(identity, expected_driver_version="595.91.07")
    with pytest.raises(FrozenInstanceError):
        identity.vendor = "Mesa"


@pytest.mark.parametrize(
    "renderer", [b"llvmpipe (LLVM 15.0.7)", b"SOFTPIPE", b"swrast", b"Software Rasterizer"]
)
def test_software_renderers_rejected(renderer):
    identity = parse_renderer_identity(b"Mesa", renderer, b"4.5 Mesa")
    assert identity.reported_device_kind == "software"
    with pytest.raises(ValueError):
        require_nvidia_renderer(identity, expected_driver_version="595.71.05")


def test_mesa_vendor_alone_does_not_imply_software():
    identity = parse_renderer_identity(b"Mesa", b"AMD Radeon", b"4.6 Mesa")
    assert identity.reported_device_kind == "unknown"


@pytest.mark.parametrize(
    "raw", [None, "NVIDIA", b"", b" ", b"x" * 513, b"a\x00b", b"a\nb", b"a\x7fb", b"\xff"]
)
@pytest.mark.parametrize("index", range(3))
def test_invalid_context_strings_rejected(raw, index):
    values = [b"NVIDIA Corporation", b"NVIDIA RTX A6000", b"4.6.0 NVIDIA 595.71.05"]
    values[index] = raw
    with pytest.raises(ValueError):
        parse_renderer_identity(*values)


def test_unrecognized_nvidia_version_is_not_certified():
    identity = parse_renderer_identity(b"NVIDIA Corporation", b"NVIDIA RTX A6000", b"unknown")
    assert identity.reported_device_kind == "unknown"


def test_forged_classification_rejected():
    forged = replace(nvidia(), renderer="llvmpipe", reported_device_kind="nvidia_gpu")
    with pytest.raises(ValueError):
        require_nvidia_renderer(forged, expected_driver_version="595.71.05")


@pytest.mark.parametrize("field", ["vendor", "renderer", "version"])
def test_forged_nontext_identity_rejected(field):
    with pytest.raises(ValueError):
        require_nvidia_renderer(
            replace(nvidia(), **{field: None}), expected_driver_version="595.71.05"
        )


@pytest.mark.parametrize("version", [None, 595, "", "595.71", "595.71.05\n"])
def test_invalid_expected_versions_rejected(version):
    with pytest.raises(ValueError):
        require_nvidia_renderer(nvidia(), expected_driver_version=version)


def test_inspection_only_queries_existing_context(monkeypatch):
    calls = []
    values = {
        1: b"NVIDIA Corporation",
        2: b"NVIDIA RTX A6000/PCIe/SSE2",
        3: b"4.6.0 NVIDIA 595.71.05",
    }

    def query(key):
        calls.append(key)
        return values[key]

    gl = SimpleNamespace(GL_VENDOR=1, GL_RENDERER=2, GL_VERSION=3, glGetString=query)
    monkeypatch.setitem(sys.modules, "OpenGL", SimpleNamespace(GL=gl))
    assert inspect_current_renderer() == nvidia()
    assert calls == [1, 2, 3]
