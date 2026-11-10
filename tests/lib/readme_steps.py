#!/usr/bin/env python3
"""Extract the ```bash blocks under the given README headings, verbatim.

usage: readme_steps.py [--harness] [--from MARK] [--to MARK] <README.md> "<heading substring>" [...]
Prints the commands of every bash block that follows a matching heading
(until the next heading), in order. Used by tests/lib/examples.sh to run the
README steps exactly as a customer would.

--from / --to keep only the lines from the line starting with MARK up to (not
including) the line starting with the --to MARK; exports before --from are kept.

--harness applies only the edits needed to run the steps unattended:
  * `kubectl logs -f job/X -n NS` -> wait for job/X to complete, then tail its log
    (timeout: $STEP_TIMEOUT, default 90m)
  * `export IMAGE=...` is kept only if IMAGE is unset (the harness picks the tag)
  * `envsubst ... < <training manifest> | kubectl apply` reads the manifest through
    $SMOKE_FILTER (if set) -- the documented smoke edit, see smoke_edit.py
"""
import re
import sys

args = sys.argv[1:]
harness = args[:1] == ['--harness']
if harness:
    args = args[1:]
start_at = stop_at = None
while args[:1] in (['--from'], ['--to']):
    if args[0] == '--from':
        start_at = args[1]
    else:
        stop_at = args[1]
    args = args[2:]
path, wanted = args[0], args[1:]
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

if start_at is not None:
    idx = next((i for i, ln in enumerate(out) if ln.lstrip().startswith(start_at)), None)
    if idx is None:
        sys.exit(f'readme_steps.py: --from marker not found: {start_at!r}')
    # keep exports from before the marker (NAMESPACE / IMAGE), skip the other commands
    out = [ln for ln in out[:idx] if ln.lstrip().startswith('export ')] + out[idx:]
if stop_at is not None:
    idx = next((i for i, ln in enumerate(out) if ln.lstrip().startswith(stop_at)), None)
    if idx is None:
        sys.exit(f'readme_steps.py: --to marker not found: {stop_at!r}')
    out = out[:idx]

if harness:
    # join `envsubst '...' \` + `  < manifest | kubectl apply -f -` continuation lines first
    joined = []
    for ln in out:
        if joined and joined[-1].rstrip().endswith('\\') and joined[-1].lstrip().startswith('envsubst ') \
                and ln.lstrip().startswith('< '):
            joined[-1] = joined[-1].rstrip()[:-1].rstrip() + ' ' + ln.strip()
        else:
            joined.append(ln)
    out = joined
    res = []
    for ln in out:
        m = re.match(r'\s*kubectl logs -f (?:-n "?\$\w+"? )?job/([\w-]+)', ln)
        if m:
            job = m.group(1)
            res.append(f'kubectl wait -n "$NAMESPACE" job/{job} --for=condition=complete '
                       f'--timeout="${{STEP_TIMEOUT:-90m}}" && kubectl logs -n "$NAMESPACE" job/{job} | tail -5')
            continue
        if re.match(r'\s*export IMAGE=', ln):
            res.append('[ -n "${IMAGE:-}" ] || ' + ln.strip())
            continue
        m = re.match(r"(\s*envsubst '[^']*' )< (\S+/manifests/[\w-]+\.yaml)( \| kubectl apply -f -.*)", ln)
        if m and m.group(2).endswith(tuple(f'/{s}.yaml' for s in (
                'maniskill-openvlaoft-ppo', 'libero-pi0-ppo', 'wan-openvlaoft-grpo', 'isaaclab-gr00t-ppo',
                'libero-pi0fast-grpo'))):
            res.append(f'{m.group(1)}< <(${{SMOKE_FILTER:-cat}} {m.group(2)}){m.group(3)}')
            continue
        res.append(ln)
    out = res
print('\n'.join(out))
