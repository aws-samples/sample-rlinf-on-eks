#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Per-phase GPU efficiency for an RLinf embodied run, from its log + DCGM in Amazon Managed Prometheus.

Reads the RLinf metric tables (one per global step) from a training log, rebuilds
each step's phase windows (sync_weights -> generate_rollouts -> cal_adv_and_returns
-> actor_training -> eval), then queries DCGM metrics for the node over each window.

Wall-clock anchor: if the Job prefixes each log line with a UTC timestamp (the
isaaclab-gr00t-ppo manifest does), each step is anchored to the line printing its
metric table. For older logs without timestamps, pass --end, the UTC time the last
table was printed (e.g. the FSx log mtime); steps are then laid out backwards from it
using RLinf's per-step "step" timer, which is only approximate.
Requires requests (plus boto3 and AMP query permission for --amp-workspace).

Usage:
    python gpu_profile.py --log train.log --host ip-10-0-0-1.<region>.compute.internal \
        --amp-workspace ws-xxxx --amp-region <region> [--profile <aws profile>] [--end <UTC>]
    # in-cluster kube-prometheus-stack instead of AMP:
    kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090 &
    python gpu_profile.py --log train.log --host <node> --prometheus-url http://localhost:9090
    python gpu_profile.py --log train.log            # timers only, no DCGM
"""
import argparse
import os
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone

PHASES = ["sync_weights", "generate_rollouts", "cal_adv_and_returns", "actor_training", "eval"]
METRICS = {
    "util%": "DCGM_FI_DEV_GPU_UTIL",
    "sm_active": "DCGM_FI_PROF_SM_ACTIVE",
    "tensor_active": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "fb_used_GiB": "DCGM_FI_DEV_FB_USED",
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_steps(path):
    """Return a list of {step, step_time, end?, <phase>: seconds} from RLinf metric tables.

    If log lines carry a leading UTC timestamp (the ts-prefix in the Job command), each
    step's `end` is the timestamp of the line that prints its metric table.
    """
    text = ANSI.sub("", open(path, errors="replace").read())
    ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) .*Metric Table", re.M)
    table_ends = [datetime.fromisoformat(m.group(1).replace("Z", "+00:00")) for m in ts_re.finditer(text)]
    steps = []
    for i, block in enumerate(text.split("Metric Table")[1:]):
        m = re.search(r"Global Step:\s+(\d+)/\d+", block)
        st = re.search(r"Step Time:\s+([\d.]+)s", block)
        if not (m and st):
            continue
        rec = {"step": int(m.group(1)), "step_time": float(st.group(1))}
        time_section = block.split("Environment")[0]
        timer = re.search(r"(?<![\w/])step=([\d.]+)", time_section)
        if timer:  # RLinf's "step" timer (includes eval); the header "Step Time" does not
            rec["step_time"] = float(timer.group(1))
        if len(table_ends) == len(text.split("Metric Table")) - 1:
            rec["end"] = table_ends[i]
        for ph in PHASES:
            v = re.search(rf"(?<![\w/]){ph}=([\d.]+)", time_section)
            if v:
                rec[ph] = float(v.group(1))
        steps.append(rec)
    return steps


def step_span(s):
    return s["step_time"]


def phase_windows(steps, start=None, end=None):
    """Phase windows in runner order within each step.

    Per-step anchors (`end` from timestamped logs) win; otherwise steps are laid out
    back-to-back from `start`, or backwards from `end`.
    """
    windows = []
    if start is None and end is not None:
        start = end - timedelta(seconds=sum(step_span(s) for s in steps))
    t = start
    for s in steps:
        if "end" in s:
            t = s["end"] - timedelta(seconds=step_span(s))
        cur = t
        for ph in PHASES:
            if ph in s:
                windows.append((s["step"], ph, cur, cur + timedelta(seconds=s[ph])))
                cur += timedelta(seconds=s[ph])
        t += timedelta(seconds=step_span(s))
    return windows


def prom_client(url):
    """Plain Prometheus HTTP API (e.g. a kubectl port-forward to kube-prometheus-stack)."""
    import requests

    base = url.rstrip("/") + "/api/v1/query_range"

    def query_range(expr, start, end, step="15s"):
        resp = requests.post(base, data={"query": expr, "start": start.isoformat(), "end": end.isoformat(),
                                         "step": step}, timeout=90)
        resp.raise_for_status()
        return resp.json()["data"]["result"]

    return query_range


def amp_client(workspace, region, profile):
    import boto3
    import requests
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from urllib.parse import urlencode

    session = boto3.Session(profile_name=profile, region_name=region)
    base = f"https://aps-workspaces.{region}.amazonaws.com/workspaces/{workspace}/api/v1/query_range"

    def query_range(expr, start, end, step="15s"):
        body = urlencode({"query": expr, "start": start.isoformat(), "end": end.isoformat(), "step": step})
        req = AWSRequest(method="POST", url=base, data=body,
                         headers={"Content-Type": "application/x-www-form-urlencoded"})
        SigV4Auth(session.get_credentials(), "aps", region).add_auth(req)
        resp = requests.post(base, data=body, headers=dict(req.headers), timeout=90)
        resp.raise_for_status()
        return resp.json()["data"]["result"]

    return query_range


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--host", action="append", default=[], help="DCGM Hostname (node name); repeat or comma-separate, optionally label=node, e.g. trainer=ip-a,sim=ip-b,sim=ip-c")
    ap.add_argument("--end", help="only for logs without line timestamps: UTC time of the last metric table")
    ap.add_argument("--amp-workspace")
    ap.add_argument("--prometheus-url", help="plain Prometheus base URL instead of AMP, e.g. http://localhost:9090")
    ap.add_argument("--amp-region", default=os.environ.get("AWS_REGION"), help="AMP workspace region (default: $AWS_REGION)")
    ap.add_argument("--profile")
    a = ap.parse_args()

    steps = parse_steps(a.log)
    if not steps:
        sys.exit("no RLinf metric tables found in the log")

    print("## Phase timers (s)\n")
    print("| step | step_time | " + " | ".join(PHASES) + " |")
    print("|---" * (len(PHASES) + 2) + "|")
    for s in steps:
        print(f"| {s['step']} | {s['step_time']:.0f} | " + " | ".join(f"{s.get(p, 0):.0f}" for p in PHASES) + " |")
    total = sum(step_span(s) for s in steps)
    print("\nshare of total: " + ", ".join(
        f"{p} {100 * sum(s.get(p, 0) for s in steps) / total:.0f}%" for p in PHASES))

    anchored = all("end" in s for s in steps)
    if anchored:
        # Per-line timestamps come from the pipe, so buffered stdout can print several
        # metric tables at once with the same timestamp. Trust per-step anchors only if
        # consecutive ones are at least half a step apart; otherwise use only the last
        # (end of run) and lay steps out backwards from it.
        gaps_ok = all((b["end"] - a["end"]).total_seconds() >= 0.5 * b["step_time"] for a, b in zip(steps, steps[1:]))
        if not gaps_ok:
            last = steps[-1]["end"]
            for s in steps:
                s.pop("end")
            anchored = False
            a.end = a.end or last.strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"\n(metric tables were flushed together; anchoring steps backwards from {a.end})")
    if not (a.host and (a.amp_workspace or a.prometheus_url) and (anchored or a.end)):
        if a.host and not anchored and not a.end:
            print("\n(no per-line timestamps in the log; pass --end to anchor DCGM windows)")
        return
    end = None if anchored else datetime.fromisoformat(a.end.replace("Z", "+00:00")).astimezone(timezone.utc)
    query_range = prom_client(a.prometheus_url) if a.prometheus_url else amp_client(a.amp_workspace, a.amp_region, a.profile)
    # --host may be repeated or comma-separated; "label=node" names a group (e.g. trainer=ip-..,sim=ip-..)
    groups = {}
    for spec in [h for arg in a.host for h in arg.split(",") if h]:
        label, _, node = spec.rpartition("=")
        groups.setdefault(label or "node", []).append(node)
    for label, nodes in groups.items():
        hosts = "|".join(nodes)
        by_phase = {p: {k: [] for k in METRICS} for p in PHASES}
        for _, ph, ws, we in phase_windows(steps, end=end):
            if (we - ws).total_seconds() < 30:
                continue
            for k, metric in METRICS.items():
                # max over duplicate exporter series per GPU, then mean across the group's GPUs
                # dcgm-exporter 3.x labels the node "Hostname", 4.x "hostname"
                expr = (f'avg(max by (Hostname, hostname, gpu) ({metric}{{Hostname=~"{hosts}"}}'
                        f' or {metric}{{hostname=~"{hosts}"}}))')
                for series in query_range(expr, ws, we):
                    vals = [float(v) for _, v in series["values"]]
                    by_phase[ph][k] += [v / 1024 for v in vals] if k == "fb_used_GiB" else vals

        print(f"\n## DCGM per phase: {label} ({len(nodes)} node(s); mean across GPUs; mean / p50)\n")
        print("| phase | " + " | ".join(METRICS) + " |")
        print("|---" * (len(METRICS) + 1) + "|")
        for ph in PHASES:
            cells = []
            for k in METRICS:
                v = by_phase[ph][k]
                cells.append(f"{statistics.mean(v):.2f} / {statistics.median(v):.2f}" if v else "-")
            if any(c != "-" for c in cells):
                print(f"| {ph} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
