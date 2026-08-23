"""Stable, bounded hashing for content-bound submission upload intents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


UPLOAD_INTENT_VERSION = "dradar-submission-upload-v2"
CHECKPOINT_V2_UPLOAD_INTENT_VERSION = (
    "dradar-checkpoint-v2-submission-upload-v1"
)
_HASH_CHUNK_BYTES = 1024 * 1024


def canonical_meta_bytes(meta: dict) -> bytes:
    return json.dumps(
        meta, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")


def _bytes_fact(value: bytes | None) -> dict:
    if value is None:
        return {"present": False}
    return {
        "present": True,
        "size": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def _path_fact(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"present": False}
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return {"present": True, "size": size, "sha256": digest.hexdigest()}


def submission_payload_manifest(
    *,
    assignment_id: str,
    session_id: str,
    resume_generation: int,
    outcome: str,
    meta: dict,
    patch: Path,
    trajectory: Path | None,
    result: Path | None,
    trajectory_bundle: Path | None,
) -> dict:
    """Describe every digest component without loading artifacts into RAM."""
    return {
        "version": UPLOAD_INTENT_VERSION,
        "assignment_id": assignment_id,
        "session_id": session_id,
        "resume_generation": resume_generation,
        "outcome": outcome,
        "components": {
            "client_meta": _bytes_fact(canonical_meta_bytes(meta)),
            "model.patch": _path_fact(patch),
            "trajectory.json": _path_fact(trajectory),
            "result.json": _path_fact(result),
            "trajectory_bundle.json": _path_fact(trajectory_bundle),
        },
    }


def upload_intent_id(manifest: dict) -> str:
    payload = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def submission_payload_sha256(**kwargs) -> str:
    """Compatibility wrapper returning the id for a freshly built manifest."""
    return upload_intent_id(submission_payload_manifest(**kwargs))


def checkpoint_v2_submission_payload_manifest(
    *,
    assignment_id: str,
    session_id: str,
    owner_epoch: int,
    outcome: str,
    meta: dict,
    patch: Path,
    trajectory: Path | None,
    result: Path | None,
    trajectory_bundle: Path | None,
) -> dict:
    return {
        "version": CHECKPOINT_V2_UPLOAD_INTENT_VERSION,
        "checkpoint_protocol_version": 2,
        "assignment_id": assignment_id,
        "session_id": session_id,
        "owner_epoch": owner_epoch,
        "outcome": outcome,
        "components": {
            "client_meta": _bytes_fact(canonical_meta_bytes(meta)),
            "model.patch": _path_fact(patch),
            "trajectory.json": _path_fact(trajectory),
            "result.json": _path_fact(result),
            "trajectory_bundle.json": _path_fact(trajectory_bundle),
        },
    }


def checkpoint_v2_submission_payload_sha256(**kwargs) -> str:
    return upload_intent_id(
        checkpoint_v2_submission_payload_manifest(**kwargs)
    )
