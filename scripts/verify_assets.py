"""Read-only file metadata check. Does not load models or hash large weights."""
import argparse, json
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--models',type=Path,required=True)
args=parser.parse_args()
manifest=json.loads((Path(__file__).resolve().parents[1]/'MODEL_ASSET_MANIFEST.json').read_text(encoding='utf-8'))
rows=[]
for item in manifest['assets']:
    p=args.models/item['path']
    size=p.stat().st_size if p.is_file() else 0
    ok=size>0 and ('bytes_observed' not in item or size==item['bytes_observed'])
    rows.append({'path':item['path'],'bytes':size,'metadata_ok':ok})
print(json.dumps({'metadata_only':True,'assets':rows},ensure_ascii=False,indent=2))
raise SystemExit(0 if all(x['metadata_ok'] for x in rows) else 1)
