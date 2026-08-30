"""Interpret the server's authoritative runner state with old-server fallback."""


_STATES = {"running", "paused", "resumable", "stale", "waiting"}


def assignment_state(assignment: dict) -> str:
    state = assignment.get("runner_state")
    if state in {"resumable", "checkpoint_retired"}:
        # ``resumable`` is a legacy server value from the retired checkpoint
        # feature.  A checkpoint_id is now only a database/protocol tombstone;
        # it must never make the CLI promise that work can be resumed.
        return "stale"
    if state in _STATES:
        return state
    if "heartbeat_running" in assignment:
        if assignment.get("heartbeat_running"):
            return "running"
        if assignment.get("execution_state") == "paused":
            return "paused"
        if assignment.get("started_at"):
            # Historical checkpoint metadata is fail-closed.  It prevents a
            # duplicate run elsewhere, but is not an executable recovery path.
            return "stale"
        return "waiting"
    # Compatibility with servers predating runner health in /assignment.
    return "running" if assignment.get("started_at") else "waiting"


def state_summary(assignments: list[dict]) -> str:
    counts = {
        state: 0
        for state in ("running", "paused", "resumable", "stale", "waiting")
    }
    for assignment in assignments:
        counts[assignment_state(assignment)] += 1
    return ", ".join(
        f"{counts[state]} {state}" for state in counts if counts[state]
    )
