from datetime import datetime, timezone

def current_timestamp() -> str:
    """
    Return the current time as an ISO‑8601 timestamp string in UTC.

    Example
    -------
    >>> ts = current_timestamp()
    >>> print(ts)          # e.g. 2025-05-16T07:12:34.567Z
    """
    # datetime.utcnow() gives naive UTC time; add tzinfo to make it explicit
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]