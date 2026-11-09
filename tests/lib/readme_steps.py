#!/usr/bin/env python3
"""Extract the ```bash blocks under the given README headings, verbatim.

usage: readme_steps.py <README.md> "<heading substring>" [...]
Prints the commands of every bash block that follows a matching '###' heading
(until the next '###'), in order. Used by tests/lib/examples.sh to run the
README steps exactly as a customer would.
"""
import re, sys

path, wanted = sys.argv[1], sys.argv[2:]
lines = open(path).read().split('\n')
out, section, inblock, take = [], None, False, False
for ln in lines:
    if ln.startswith('#') and not inblock:
        section = ln
        take = any(w in ln for w in wanted)
        continue
    if ln.strip().startswith('```'):
        if not inblock:
            inblock = ln.strip() in ('```bash', '```sh')
            if inblock and take:
                out.append(f'# --- {section.strip()}')
            continue
        inblock = False
        continue
    if inblock and take:
        out.append(ln)
print('\n'.join(out))
