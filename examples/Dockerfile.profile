# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# =============================================================================
# RLinf image profile (slim stage 1)
#
# Builds only the venvs a profile needs, on top of upstream's
# `embodied-common-image` (system deps, uv, RLinf requirements/, switch_env,
# download_assets/link_assets; no venvs). The result is then layered with the
# EFA overlay by examples/Dockerfile, exactly like the full image.
#
# Used by examples/buildspec.yml when PROFILE != full. Profiles are defined in
# examples/profiles.env. Local build (from this repo root, after cloning RLinf/):
#   docker build --target embodied-common-image \
#     --build-arg BUILD_TARGET=embodied-maniskill_libero --build-arg NO_MIRROR=1 \
#     -t rlinf-upstream-embodied-common -f RLinf/docker/Dockerfile RLinf/
#   docker build -f examples/Dockerfile.profile \
#     --build-arg INSTALLS="openpi:openpi:maniskill_libero" \
#     --build-arg ASSETS=maniskill,openpi --build-arg ASSET_VENV=openpi \
#     -t rlinf-upstream-profile-libero-pi0 .
#   docker build --build-arg BUILD_TARGET=profile-libero-pi0 -f examples/Dockerfile .
# =============================================================================

ARG BASE=rlinf-upstream-embodied-common
FROM ${BASE}

# Space-separated venv:model:env tuples, passed to upstream requirements/install.sh.
ARG INSTALLS
# Comma-separated download_assets list (e.g. "maniskill,openpi"); empty = none.
ARG ASSETS=""
# Venv used to run download_assets; must be one of INSTALLS and have mani_skill + hf.
ARG ASSET_VENV=""

SHELL ["/bin/bash", "-c"]
WORKDIR ${UV_PATH}

RUN test -n "${INSTALLS}" || { echo "ERROR: INSTALLS is empty"; exit 1; }; \
    test -f requirements/install.sh || { echo "ERROR: ${UV_PATH}/requirements/install.sh missing; is BASE upstream embodied-common-image?"; exit 1; }; \
    for t in ${INSTALLS}; do \
        IFS=: read -r v m e <<<"$t"; \
        { [ -n "$v" ] && [ -n "$m" ] && [ -n "$e" ]; } || { echo "ERROR: bad INSTALLS tuple '$t' (want venv:model:env)"; exit 1; }; \
        echo "=== install.sh embodied --venv $v --model $m --env $e ==="; \
        bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} \
            embodied --venv "$v" --model "$m" --env "$e" || exit 1; \
    done

RUN if [ -n "${ASSETS}" ]; then \
        test -n "${ASSET_VENV}" || { echo "ERROR: ASSETS set but ASSET_VENV empty"; exit 1; }; \
        source switch_env "${ASSET_VENV}" && \
        download_assets ${INSTALL_MIRROR_OPTION} --dir /opt/assets --assets "${ASSETS}" && \
        link_assets; \
    fi

# Mirror upstream final-image cleanup, and record which venvs this profile ships
# (read by the stage-2 DreamZero guard and the L2 container test).
RUN uv cache prune && rm -rf /var/lib/apt/lists/* && \
    for t in ${INSTALLS}; do echo "${t%%:*}"; done > "${UV_PATH}/.profile" && \
    echo "Profile venvs:" && cat "${UV_PATH}/.profile"

WORKDIR /workspace
CMD ["/bin/bash"]
