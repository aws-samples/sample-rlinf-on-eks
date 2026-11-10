#!/usr/bin/env python3
"""Instance types in docs/diagrams must match what the manifests schedule on.

For each example: every instance type a training manifest pins (node affinity
`node.kubernetes.io/instance-type` values) must appear in the example README's
**Hardware** row and in the diagram it embeds; the gpu-sim NodePool default in
infrastructure/workloads/variables.tf must appear in each diagram that lists the pool.
Exit 1 with a list of mismatches.
"""
import glob
import html
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else '.'
os.chdir(root)
TYPE = re.compile(r'\b[a-z][a-z0-9]*\.[0-9]*x?large\b')
errors = []


def manifest_types(path):
    # non-comment lines only (comments describe alternatives, e.g. "use p5en instead")
    s = '\n'.join(ln for ln in open(path).read().split('\n') if not ln.lstrip().startswith('#'))
    out = set(TYPE.findall(' '.join(re.findall(r'node\.kubernetes\.io/instance-type:\s*(\S+)', s))))
    for m in re.finditer(r'key: node\.kubernetes\.io/instance-type\s*\n\s*operator: In\s*\n\s*values:\s*((?:\n\s*- .*)+|\[.*\])', s):
        out |= set(TYPE.findall(m.group(1)))
    return out


def diagram_text(readme):
    txt = ''
    for ref in re.findall(r'\]\(([^)]+\.drawio)\.svg\)', open(readme).read()):
        p = os.path.normpath(os.path.join(os.path.dirname(readme), ref))
        if os.path.exists(p):
            txt += html.unescape(open(p).read())
    return txt


def expand(text):
    """g7e.12/24/48xlarge -> g7e.12xlarge g7e.24xlarge g7e.48xlarge"""
    def rep(m):
        fam, sizes = m.group(1), m.group(2).split('/')
        return ' '.join(f'{fam}.{z}xlarge' for z in sizes)
    text = re.sub(r'\b([a-z][a-z0-9]*)\.((?:[0-9]+/)+[0-9]+)xlarge', rep, text)
    # g7e/g6e.48xlarge -> g7e.48xlarge g6e.48xlarge
    return re.sub(r'\b((?:[a-z][a-z0-9]*/)+[a-z][a-z0-9]*)\.([0-9]*x?large)\b',
                  lambda m: ' '.join(f'{f}.{m.group(2)}' for f in m.group(1).split('/')), text)


for readme in sorted(glob.glob('examples/*/README.md')):
    ex = os.path.dirname(readme)
    pinned = set()
    for m in glob.glob(f'{ex}/manifests/*.yaml'):
        if os.path.basename(m) in ('model-download.yaml', 'stage-isaacsim.yaml', 'generate-metadata.yaml'):
            continue
        pinned |= manifest_types(m)
    if not pinned:
        continue
    text = open(readme).read()
    hw = re.search(r'^\|\s*\*\*Hardware\*\*\s*\|(.*)$', text, re.M)
    hw_types = set(TYPE.findall(expand(hw.group(1)))) if hw else set(TYPE.findall(expand(text)))
    for t in sorted(pinned - hw_types):
        errors.append(f'{readme}: manifests schedule on {t}, not in the Hardware row')
    dia = expand(diagram_text(readme))
    if dia:
        for t in sorted(pinned - set(TYPE.findall(dia))):
            errors.append(f'{readme}: manifests schedule on {t}, not in its diagram')

var = open('infrastructure/workloads/variables.tf').read()
m = re.search(r'variable "sim_instance_types".*?default\s*=\s*\[(.*?)\]', var, re.S)
sim = set(TYPE.findall(m.group(1))) if m else set()
for d in glob.glob('examples/**/*.drawio', recursive=True):
    t = expand(html.unescape(open(d).read()))
    if 'gpu-sim NodePool' in t:
        for x in sorted(sim - set(TYPE.findall(t))):
            errors.append(f'{d}: gpu-sim NodePool default includes {x}, diagram does not')

for e in errors:
    print(e)
sys.exit(1 if errors else 0)
