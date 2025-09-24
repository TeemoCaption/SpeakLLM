from pathlib import Path
path = Path('speechllm/models/speechllm.py')
for i,line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
    if 1168 <= i <= 1182:
        print(f"{i:03}: {line!r}")
