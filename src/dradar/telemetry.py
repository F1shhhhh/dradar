"""Low-bandwidth runner heartbeat lifecycle.

One session reports regardless of how many cells it holds.  Payloads contain
only lifecycle metadata; never task text, prompts, trajectories, patches,
commands, hostname, username, IP or hardware details.
"""

from __future__ import annotations

import random
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from . import __version__
from .api_client import ApiClient, ApiError
from .checkpoint_observation_v2 import CheckpointObservationReporterV2


def platform_family() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
            return "wsl"
        return "linux"
    return "other"


class RunnerTelemetry:
    """A daemon heartbeat with adaptive 60/120-second cadence.

    Telemetry is best effort and can never abort a trial.  Three consecutive
    failures produce one warning so a user knows the server can no longer see
    their runner; recovery produces one matching notice.
    """

    def __init__(
        self,
        client: ApiClient,
        *,
        jitter: bool = True,
        target_workers: int = 1,
    ):
        if not 1 <= target_workers <= 40:
            raise ValueError("target_workers must be between 1 and 40")
        self.client = client
        self.target_workers = target_workers
        self.session_id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = "preparing"
        self._active_assignment_id: str | None = None
        self._resume_generation: int | None = None
        self._batch_id: str | None = None
        self._seq = 0
        self._progress_counter = 0
        self._interval = 120
        self._failures = 0
        self._warned = False
        self._disabled = False
        self._jitter = jitter
        self._shown_notice_ids: set[str] = set()
        self._checkpoint_container_backend: str | None = None
        self._checkpoint_machine_fingerprint: str | None = None
        self._checkpoint_observation_reporter: (
            CheckpointObservationReporterV2 | None
        ) = None

    def configure_checkpoint_runtime(
        self,
        *,
        container_backend: str,
        machine_fingerprint: str,
    ) -> None:
        """Attach immutable, privacy-preserving V2 recovery cohort facts."""

        if container_backend not in {
            "docker", "orbstack", "podman", "native", "other",
        }:
            raise ValueError("unknown checkpoint container backend")
        if (
            not isinstance(machine_fingerprint, str)
            or len(machine_fingerprint) != 64
            or any(ch not in "0123456789abcdef" for ch in machine_fingerprint)
        ):
            raise ValueError("invalid checkpoint machine fingerprint")
        with self._lock:
            current = (
                self._checkpoint_container_backend,
                self._checkpoint_machine_fingerprint,
            )
            requested = (container_backend, machine_fingerprint)
            if current != (None, None) and current != requested:
                raise ValueError("checkpoint runtime identity is immutable")
            self._checkpoint_container_backend = container_backend
            self._checkpoint_machine_fingerprint = machine_fingerprint

    def configure_checkpoint_observation_reporting(self, home: Path) -> None:
        """Enable the optional shadow outbox for a non-OFF V2 activation.

        Merely constructing ``RunnerTelemetry`` must remain a strict no-op for
        Checkpoint V2.  The rollout coordinator calls this method only after
        the assignment/server negotiation enables capture.  The reporter is
        deliberately independent from the heartbeat: a slow observation POST
        cannot consume this thread's send lock or delay an atomic checkout.
        """

        requested_root = (
            Path(home).absolute() / "checkpoint-v2" / "observations"
        )
        with self._lock:
            current = self._checkpoint_observation_reporter
            if current is not None:
                if current.spool.root != requested_root:
                    raise ValueError(
                        "checkpoint observation account boundary is immutable",
                    )
                return
            reporter = CheckpointObservationReporterV2(self.client, Path(home))
            self._checkpoint_observation_reporter = reporter
            already_started = self._thread is not None
        if already_started:
            reporter.start()

    def record_checkpoint_observation(self, payload: dict) -> bool:
        """Queue shadow evidence without I/O, waiting, or raising."""

        with self._lock:
            reporter = self._checkpoint_observation_reporter
        if reporter is None:
            return False
        try:
            return reporter.record(payload)
        except Exception:
            return False

    def register_checkpoint_cohort(self, payload: dict) -> bool:
        """Bind local evidence to this exact runner session and cohort."""

        with self._lock:
            reporter = self._checkpoint_observation_reporter
            session_id = self.session_id
        if reporter is None:
            return False
        materialized = dict(payload)
        materialized["runner_session_id"] = session_id
        try:
            return reporter.register_cohort(materialized)
        except Exception:
            return False

    def _show_notices(self, response: dict) -> None:
        """Print each bounded server notice at most once per runner process."""
        notices = response.get("notices")
        if not isinstance(notices, list):
            return
        for value in notices[:10]:
            if not isinstance(value, dict):
                continue
            notice_id = value.get("id")
            message = value.get("message")
            severity = value.get("severity", "info")
            if (
                not isinstance(notice_id, str)
                or not 1 <= len(notice_id) <= 100
                or not isinstance(message, str)
                or not 1 <= len(message) <= 1000
                or severity not in {"info", "warning", "critical"}
                or notice_id in self._shown_notice_ids
            ):
                continue
            self._shown_notice_ids.add(notice_id)
            clean = " ".join(message.splitlines())
            print(f"server notice [{severity}]: {clean}", file=sys.stderr)

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._checkpoint_observation_reporter is not None:
            self._checkpoint_observation_reporter.start()
        self._thread = threading.Thread(
            target=self._loop, name="dradar-heartbeat", daemon=True)
        self._thread.start()

    def bind_batch(self, batch_id: str | None) -> None:
        if not batch_id:
            return
        with self._lock:
            changed = self._batch_id != batch_id
            self._batch_id = batch_id
            if changed:
                self._progress_counter += 1
        if changed:
            self._wake.set()

    def set_phase(
        self,
        phase: str,
        assignment_id: str | None = None,
        resume_generation: int | None = None,
    ) -> None:
        if phase not in {"preparing", "queued", "running", "uploading", "paused"}:
            raise ValueError(f"unknown runner phase {phase!r}")
        if resume_generation is not None and resume_generation < 0:
            raise ValueError("resume_generation must be non-negative")
        if assignment_id is None:
            resume_generation = None
        with self._lock:
            changed = (
                self._phase, self._active_assignment_id, self._resume_generation,
            ) != (phase, assignment_id, resume_generation)
            self._phase = phase
            self._active_assignment_id = assignment_id
            self._resume_generation = resume_generation
            if changed:
                self._progress_counter += 1
        if changed:
            self._wake.set()

    def _payload(self) -> dict:
        with self._lock:
            self._seq += 1
            payload = {
                "protocol_version": 2,
                "client_version": __version__,
                "session_id": self.session_id,
                "batch_id": self._batch_id,
                "seq": self._seq,
                "phase": self._phase,
                "active_assignment_id": self._active_assignment_id,
                "resume_generation": self._resume_generation,
                "client_monotonic_ms": int(time.monotonic() * 1000),
                "progress_counter": self._progress_counter,
                "platform": platform_family(),
                "target_workers": self.target_workers,
            }
            if self._checkpoint_container_backend is not None:
                payload["container_backend"] = self._checkpoint_container_backend
                payload["checkpoint_machine_fingerprint"] = (
                    self._checkpoint_machine_fingerprint
                )
            return payload

    def _send_once(self) -> int:
        """Send once and return the server-selected next interval."""
        with self._send_lock:
            if self._disabled:
                return self._interval
            try:
                response = self.client.runner_heartbeat(self._payload())
            except ApiError as exc:
                # Older servers have no endpoint. Silence and disable rather than
                # alarming users or producing a 404 every two minutes forever.
                if exc.status_code == 404:
                    self._disabled = True
                    return self._interval
                self._failures += 1
                if self._failures >= 3 and not self._warned:
                    print("warning: the server cannot see this runner's heartbeat; "
                          "work continues and your leases are not auto-released",
                          file=sys.stderr)
                    self._warned = True
                return self._interval
            except Exception:
                self._failures += 1
                if self._failures >= 3 and not self._warned:
                    print("warning: runner heartbeat is unavailable; work continues and "
                          "your leases are not auto-released", file=sys.stderr)
                    self._warned = True
                return self._interval

            if self._warned:
                print("runner heartbeat recovered", file=sys.stderr)
            self._failures = 0
            self._warned = False
            self._show_notices(response)
            if response.get("batch_id"):
                with self._lock:
                    self._batch_id = response["batch_id"]
            requested = response.get("next_heartbeat_sec", self._interval)
            try:
                self._interval = min(600, max(30, int(requested)))
            except (TypeError, ValueError):
                pass
            return self._interval

    def flush(self) -> None:
        """Synchronously register the latest state before an atomic checkout."""
        self._send_once()

    def _loop(self) -> None:
        while not self._stop.is_set():
            interval = self._send_once()
            if self._jitter:
                interval *= random.uniform(0.9, 1.1)
            self._wake.wait(interval)
            self._wake.clear()

    def close(self, reason: str) -> None:
        if reason not in {"completed", "paused", "interrupted", "error"}:
            raise ValueError(f"unknown close reason {reason!r}")
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._checkpoint_observation_reporter is not None:
            self._checkpoint_observation_reporter.close(timeout=0.25)
        if self._disabled:
            return
        with self._lock:
            self._seq += 1
            payload = {
                "session_id": self.session_id,
                "batch_id": self._batch_id,
                "seq": self._seq,
                "reason": reason,
            }
        with self._send_lock:
            try:
                self.client.runner_close(payload)
            except Exception:
                pass
