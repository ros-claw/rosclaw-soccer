"""Reusable offline native/reference bundle allocation for bounded SIM studies.

Callers pin sources/configuration, admit resources, own the journal and persist
all failures. No transport, runtime executor, optimizer or policy promotion.
"""

import gc
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NativeFirstTouchBundle:
    factory: Any
    compiler: Any
    auditor: Any
    reader: Any
    encoder_session: Any


def allocate_native_first_touch_bundle(assets, reference_receipt, root, *, record_phase):
    import onnxruntime as ort
    import torch

    from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
    from rosclaw_soccer.rsi.foundation_torch_capture_audit import TorchCallAuditor
    from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
    from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
        RecurrentSamplingEpisodeFactory,
    )
    from rosclaw_soccer.rsi.sampling_reference_audit import SamplingReferenceAuditCompiler
    from scripts.rsi_mesh_contact_native_pilot import load_historical_report

    torch.set_num_threads(1)
    record_phase("VERIFYING_OWNED_HISTORICAL_REPORT")
    historical = load_historical_report(reference_receipt)
    record_phase("ALLOCATING_INDEPENDENT_REFERENCE")
    compiler = SamplingReferenceAuditCompiler(historical["executed_motor_policy"])
    del historical
    gc.collect()
    record_phase("ALLOCATING_FIXED_MEAN")
    mean = load_json_artifact(assets.EXPORT / "model-0.json.gz")
    factory = RecurrentSamplingEpisodeFactory(mean)
    del mean
    gc.collect()
    record_phase("ALLOCATING_EPISODE_OWNED_PAYLOAD_READER")
    shared = root / ".shared-models"
    shared.mkdir()
    witness = reference_receipt["witnesses"]["proof_payload"]
    original = Path(witness["path"])
    copied = shared / original.name
    shutil.copyfile(original, copied)
    reader = FixedPhysicalPayloadReader(copied, expected_file_hash=witness["sha256"])
    record_phase("ALLOCATING_FOUNDATION_AUDITOR_AND_ENCODER")
    auditor = TorchCallAuditor(assets.SONIC)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(assets.SONIC / "planner_sonic.onnx"), options, providers=["CPUExecutionProvider"]
    )
    record_phase("FIXED_NATIVE_REFERENCE_BUNDLE_READY")
    return NativeFirstTouchBundle(factory, compiler, auditor, reader, session)
