#!/usr/bin/env python3
"""Relative Markdown links and images must point at files that exist, and exported
diagrams (*.drawio.svg) must have a solid background.

usage: link_check.py <repo root>. Checks every *.md outside docs/ (including
.opencode/skills/*/SKILL.md); prints broken links and exits 1 if any.
"""
import glob
import os
import re
import sys

os.chdir(sys.argv[1] if len(sys.argv) > 1 else '.')
files = [f for f in glob.glob('**/*.md', recursive=True) if not f.startswith('docs/')]
files += glob.glob('.opencode/skills/*/SKILL.md')
bad = []
for f in sorted(set(files)):
    for m in re.finditer(r'\]\(([^)#\s]+)(#[^)]*)?\)', open(f).read()):
        t = m.group(1)
        if re.match(r'^[a-z][a-z0-9+.-]*:', t):   # http:, https:, mailto:
            continue
        p = t.lstrip('/') if t.startswith('/') else os.path.normpath(os.path.join(os.path.dirname(f), t))
        if not os.path.exists(p):
            bad.append(f'{f}: {t}')
# Exported diagrams (*.drawio.svg) need a solid background: transparent ones render
# unreadable on GitHub's dark theme and look inconsistent next to the others. Set
# background="#ffffff" on the .drawio's <mxGraphModel> and re-export (no -t/--transparent).
for f in sorted(glob.glob('**/*.drawio.svg', recursive=True)):
    if f.startswith('docs/'):
        continue
    head = open(f, errors='ignore').read(4000)
    m = re.search(r'<svg[^>]*style="[^"]*background:\s*([^;"]+)', head)
    if not m or m.group(1).strip() in ('transparent', 'none'):
        bad.append(f'{f}: transparent background (re-export with background="#ffffff")')
print('\n'.join(bad))
sys.exit(1 if bad else 0)
