import hashlib
import json

import pytest

from rosclaw_soccer.rsi.queued_audit_commitment import bind_complete


def fixture(tmp_path):
    head = "a" * 40
    dp = tmp_path / "declaration.json"
    dp.write_text(
        json.dumps(dict(source_head=head, hardware_authorized=False, promotion_authorized=False))
    )
    dh = "sha256:" + hashlib.sha256(dp.read_bytes()).hexdigest()
    cp = tmp_path / "complete.json"
    cp.write_text(
        json.dumps(
            dict(
                source_head=head,
                declaration_hash=dh,
                hardware_authorized=False,
                promotion_authorized=False,
            )
        )
    )
    ch = "sha256:" + hashlib.sha256(cp.read_bytes()).hexdigest()
    return head, dh, ch


def test_queued_audit_binds_receipt_after_exact_preregistration(tmp_path):
    head, dh, ch = fixture(tmp_path)
    assert (
        bind_complete(tmp_path, complete_hash=None, producer_head=head, declaration_hash=dh) == ch
    )


def test_direct_commitment_still_supported(tmp_path):
    _, _, ch = fixture(tmp_path)
    assert (
        bind_complete(tmp_path, complete_hash=ch, producer_head=None, declaration_hash=None) == ch
    )


@pytest.mark.parametrize(
    "fault", ["head", "declaration", "failure", "missing_complete", "mixed", "real", "short_head"]
)
def test_no_audit_on_wrong_or_partial_producer(tmp_path, fault):
    head, dh, ch = fixture(tmp_path)
    complete = None
    if fault == "head":
        head = "b" * 40
    if fault == "short_head":
        head = "a" * 8
    if fault == "declaration":
        dh = "sha256:" + "0" * 64
    if fault == "failure":
        (tmp_path / "failed.json").write_text("{}")
    if fault == "missing_complete":
        (tmp_path / "complete.json").unlink()
    if fault == "mixed":
        complete = ch
    if fault == "real":
        p = tmp_path / "complete.json"
        r = json.loads(p.read_text())
        r["hardware_authorized"] = True
        p.write_text(json.dumps(r))
    with pytest.raises((ValueError, FileNotFoundError)):
        bind_complete(tmp_path, complete_hash=complete, producer_head=head, declaration_hash=dh)
