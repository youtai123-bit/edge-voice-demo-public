"""Verify archived source/evidence identities and parse Python without imports."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = json.loads((ROOT / 'SOURCE_PROVENANCE.json').read_text(encoding='utf-8'))
    for row in rows:
        path = ROOT / row['file']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise SystemExit('Source identity mismatch: ' + row['file'])
    evidence = ROOT / 'evidence'
    items = json.loads((evidence / 'SOURCE_EVIDENCE_INDEX.json').read_text(encoding='utf-8'))
    for row in items:
        path = evidence / row['file']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise SystemExit('Evidence identity mismatch: ' + row['file'])
    python_files = sorted((ROOT / 'app').glob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    for path in python_files:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path.relative_to(ROOT)))
    print(json.dumps({
        'status': 'passed', 'archived_source_identities': len(rows),
        'remote_receipt_bound_sources': sum(bool(x['matches_deployed_receipt']) for x in rows),
        'evidence_identities': len(items), 'python_files_parsed': len(python_files),
        'model_or_audio_loaded': False,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
