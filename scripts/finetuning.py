import re
import json
import matplotlib.pyplot as plt


def get_latest_checkpoint(directory):
    pattern = re.compile(r'checkpoint-(\d+)')

    max_checkpoint = -1
    latest_checkpoint_dir = None

    for entry in os.listdir(directory):
        match = pattern.match(entry)
        if match:
            checkpoint_num = int(match.group(1))
            if checkpoint_num > max_checkpoint:
                max_checkpoint = checkpoint_num
                latest_checkpoint_dir = entry

    return os.path.join(directory, latest_checkpoint_dir)


def plot_loss_from_trainer_state(json_path):
    with open(json_path, 'r') as f:
        trainer_state = json.load(f)

    steps = []
    loss_values = []
    eval_steps = []
    eval_loss_values = []

    for entry in trainer_state['log_history']:
        step = entry['step']
        if 'loss' in entry:
            if step % 8 == 0:
                steps.append(step)
                loss_values.append(entry['loss'])
        elif 'eval_loss' in entry:
            eval_steps.append(step)
            eval_loss_values.append(entry['eval_loss'])

    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_values, label='Training Loss')
    plt.plot(eval_steps, eval_loss_values, label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Steps')
    plt.legend()
    plt.grid(True)
    plt.show()


def insert_shebang(script_path: str):
  with open(script_path, 'r') as file:
    lines = file.readlines()

  if not lines[0].startswith('#!'):
      lines.insert(0, '#!/usr/bin/env python3\n')
      with open(script_path, 'w') as file:
          file.writelines(lines)


def insert_code(filename, code, line_number):
    with open(filename, 'r') as file:
        lines = file.readlines()

    lines.insert(line_number - 1, code + '\n')

    with open(filename, 'w') as file:
          file.writelines(lines)