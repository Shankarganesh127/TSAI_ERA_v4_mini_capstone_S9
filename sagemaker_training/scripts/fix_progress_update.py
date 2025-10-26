import re
p = r"d:\\TSAI\\ERAv4\\ERAv4_class_S9\\ERAv4_mini_capstone_S9\\ERAv4_mini_capstone_S9\\TSAI_ERAv4_mini_capstone_S9\\sagemaker_training\\imagenet_training_pipeline.py"
with open(p, 'r', encoding='utf-8', newline='') as f:
    lines = f.readlines()
start_idx = None
for i,l in enumerate(lines):
    if 'progress_manager.update_status' in l:
        start_idx = i
        break
if start_idx is None:
    print('No progress_manager.update_status found')
    raise SystemExit(0)
# Find end of the call by matching parentheses balance from start_idx
balance = 0
end_idx = None
for j in range(start_idx, min(start_idx+20, len(lines))):
    line = lines[j]
    for ch in line:
        if ch == '(':
            balance += 1
        elif ch == ')':
            balance -= 1
    if balance <= 0:
        end_idx = j
        break
if end_idx is None:
    print('Could not find end of update_status call in nearby lines. Aborting.')
    raise SystemExit(1)
# Replace the block with a clean multiline call
replacement = [
    '    # Training finished — report best validation accuracy\n',
    '    progress_manager.update_status(\n',
    "        full_train_key,\n",
    "        f\"[OK] STEP 6: Full Training Complete - Best Val Acc: {max(history['val_acc']):.2f}%\"\n",
    '    )\n'
]
new_lines = lines[:start_idx] + replacement + lines[end_idx+1:]
with open(p, 'w', encoding='utf-8', newline='') as f:
    f.writelines(new_lines)
print('Rewrote progress_manager.update_status block')
