from pathlib import Path

path = Path('speechllm/training/trainer.py')
lines = path.read_text(encoding='utf-8').splitlines()

start = None
end = None
for idx, line in enumerate(lines):
    if line.strip() == '# 移動到設備':
        start = idx
    if line.strip() == '# 前向傳播' and start is not None:
        end = idx
        break

if start is None or end is None:
    raise SystemExit('could not locate replacement block')

new_block = [
    '            # 移動到設備',
    '            filtered_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v',
    '                            for k, v in filtered_batch.items()}',
    '',
    '            # 過濾掉沒有有效標籤的樣本（避免 loss 出現 NaN）',
    "            if 'labels' in filtered_batch and isinstance(filtered_batch['labels'], torch.Tensor):",
    '                labels = filtered_batch["labels"]',
    '                if labels.dim() > 1:',
    '                    valid_mask = (labels != -100).any(dim=1)',
    '                else:',
    '                    valid_mask = labels != -100',
    '',
    '                if not valid_mask.any():',
    '                    continue',
    '',
    '                if not torch.all(valid_mask):',
    '                    valid_indices = valid_mask.nonzero(as_tuple=False).view(-1).tolist()',
    '                    filtered_batch = {',
    '                        key: value[valid_indices] if isinstance(value, torch.Tensor) else',
    '                        ([value[i] for i in valid_indices] if isinstance(value, list) else value)',
    '                        for key, value in filtered_batch.items()',
    '                    }',
    ''
]

lines[start:end] = new_block

# ensure NaN check directly after forward-step line
for idx, line in enumerate(lines):
    if line.strip() == '# 前向傳播':
        forward_idx = idx + 1
        break
else:
    raise SystemExit('forward step comment missing')

if lines[forward_idx + 1].strip() != 'if torch.isnan(loss):':
    lines[forward_idx + 1:forward_idx + 1] = [
        '            if torch.isnan(loss):',
        "                self.logger.warning('Encountered NaN loss; skipping this batch.')",
        '                continue',
        ''
    ]

path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
