"""Read a bounded recent tail without rewriting the original transcript archive."""
from __future__ import annotations
import json
from pathlib import Path

def recent_transcripts(path: Path, limit: int = 100, byte_limit: int = 524288) -> list[dict]:
    if not path.is_file(): return []
    rows = []
    with path.open('rb') as stream:
        stream.seek(0, 2)
        start = max(0, stream.tell() - byte_limit)
        stream.seek(start)
        if start: stream.readline()
        while True:
            offset = stream.tell()
            line = stream.readline(byte_limit + 1)
            if not line: break
            if len(line) > byte_limit: break
            try:
                row = json.loads(line)
                if not isinstance(row.get('text'), str) or not isinstance(row.get('index'), int): continue
                row.setdefault('record_id', 'archive-' + str(offset))
                row.setdefault('session_id', 'archive')
                rows.append(row)
            except (ValueError, TypeError, AttributeError): continue
    return rows[-limit:]
