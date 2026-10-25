#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Resolve an image profile for examples/buildspec.yml (bash only; CodeBuild runs bash).
# Usage: source examples/scripts/resolve_profile.sh && resolve_profile
# Inputs (env): PROFILE (default full), optional INSTALLS / ASSETS / ASSET_VENV overrides,
#               PROFILES_FILE (default examples/profiles.env).
# Outputs (exported): PROFILE, INSTALLS, ASSETS, ASSET_VENV, PROFILE_TAG_PREFIX.
# For PROFILE=full, INSTALLS stays empty and PROFILE_TAG_PREFIX=${BUILD_TARGET}.

resolve_profile() {
  local file="${PROFILES_FILE:-examples/profiles.env}"
  PROFILE="${PROFILE:-full}"

  if [[ ! "$PROFILE" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    echo "ERROR: invalid PROFILE '$PROFILE' (lowercase letters, digits, '-')" >&2
    return 1
  fi

  if [[ "$PROFILE" == "full" ]]; then
    if [[ -n "${INSTALLS:-}" ]]; then
      echo "ERROR: INSTALLS is set but PROFILE=full; name a custom profile instead" >&2
      return 1
    fi
    PROFILE_TAG_PREFIX="${BUILD_TARGET:-embodied-maniskill_libero}"
    export PROFILE PROFILE_TAG_PREFIX
    return 0
  fi

  local line="" p_installs="" p_assets="" p_asset_venv=""
  line=$(grep -v '^[[:space:]]*#' "$file" | grep -E "^${PROFILE}\|" | head -1 || true)
  if [[ -n "$line" ]]; then
    IFS='|' read -r _ p_installs p_assets p_asset_venv <<<"$line"
  elif [[ -z "${INSTALLS:-}" ]]; then
    local known
    known=$(grep -v '^[[:space:]]*#' "$file" | cut -d'|' -f1 | grep -v '^$' | tr '\n' ' ')
    echo "ERROR: unknown PROFILE '$PROFILE' and no INSTALLS given. Known: full ${known}" >&2
    return 1
  fi

  INSTALLS="${INSTALLS:-$p_installs}"
  ASSETS="${ASSETS-$p_assets}"
  ASSET_VENV="${ASSET_VENV-$p_asset_venv}"

  local t venvs=() tuples=()
  read -r -a tuples <<<"$INSTALLS"
  for t in "${tuples[@]}"; do
    if [[ ! "$t" =~ ^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$ ]]; then
      echo "ERROR: bad INSTALLS tuple '$t' (want venv:model:env)" >&2
      return 1
    fi
    venvs+=("${t%%:*}")
  done
  if [[ ${#venvs[@]} -eq 0 ]]; then
    echo "ERROR: profile '$PROFILE' has no INSTALLS" >&2
    return 1
  fi
  if [[ -n "$ASSETS" ]]; then
    if [[ -z "$ASSET_VENV" ]] || [[ ! " ${venvs[*]} " == *" ${ASSET_VENV} "* ]]; then
      echo "ERROR: ASSET_VENV '${ASSET_VENV}' must be one of the profile venvs: ${venvs[*]}" >&2
      return 1
    fi
  fi

  PROFILE_TAG_PREFIX="$PROFILE"
  export PROFILE INSTALLS ASSETS ASSET_VENV PROFILE_TAG_PREFIX
}
