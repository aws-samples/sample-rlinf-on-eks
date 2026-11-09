#!/usr/bin/env python3
"""The documented smoke edit: append overrides to a training manifest's HYDRA_OVERRIDES.

usage: smoke_edit.py <manifest.yaml> [overrides]
  overrides default: $SMOKE_EXTRA, else runner.max_steps=2 runner.save_interval=-1
Prints the edited manifest. This is the only change the harness makes to a committed
training manifest (see "For a short smoke run" in each example README).
"""
import os
import sys

path = sys.argv[1]
extra = (sys.argv[2] if len(sys.argv) > 2 else
         os.environ.get('SMOKE_EXTRA') or 'runner.max_steps=2 runner.save_interval=-1')
s = open(path).read()
i = s.index('- name: HYDRA_OVERRIDES')
j = s.index('value: "', i) + len('value: "')
k = s.index('"', j)
sys.stdout.write(s[:k] + ' ' + extra + s[k:])
