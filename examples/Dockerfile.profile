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

# Space-separated venv:model:env[:flags] tuples, passed to upstream requirements/install.sh.
# flags (optional): extra install.sh options, comma-separated, e.g. --python,3.12.12,--torch,2.11.0 (commas become spaces)
ARG INSTALLS
# Comma-separated download_assets list (e.g. "maniskill,openpi"); empty = none.
ARG ASSETS=""
# Venv used to run download_assets; must be one of INSTALLS and have mani_skill + hf.
ARG ASSET_VENV=""
# Shared IsaacLab checkout for --env isaaclab installs (matches upstream's
# embodied-isaaclab target). install.sh's clone_or_reuse_repo uses it when set.
ENV ISAAC_LAB_PATH=/opt/envs/isaaclab

SHELL ["/bin/bash", "-c"]
WORKDIR ${UV_PATH}

RUN test -n "${INSTALLS}" || { echo "ERROR: INSTALLS is empty"; exit 1; }; \
    test -f requirements/install.sh || { echo "ERROR: ${UV_PATH}/requirements/install.sh missing; is BASE upstream embodied-common-image?"; exit 1; }; \
    for t in ${INSTALLS}; do \
        IFS=: read -r v m e x <<<"$t"; \
        { [ -n "$v" ] && [ -n "$m" ] && [ -n "$e" ]; } || { echo "ERROR: bad INSTALLS tuple '$t' (want venv:model:env[:flags])"; exit 1; }; \
        x="${x//,/ }"; \
        echo "=== install.sh embodied --venv $v --model $m --env $e $x ==="; \
        # hf-libero (pi0_fast --env libero) prompts "custom path for the dataset folder? (Y/N)"
        # on its first import if ~/.libero/config.yaml is missing, and install.sh imports it from
        # a heredoc (no stdin): EOFError. A placeholder config skips the prompt; install.sh's
        # reset_libero_config then rewrites it with the venv's real paths.
        if [[ $e == *libero* ]]; then mkdir -p ~/.libero && { [ -f ~/.libero/config.yaml ] || echo "benchmark_root: /tmp" > ~/.libero/config.yaml; }; fi; \
        bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} \
            embodied --venv "$v" --model "$m" --env "$e" $x || exit 1; \
    done

RUN if [ -n "${ASSETS}" ]; then \
        test -n "${ASSET_VENV}" || { echo "ERROR: ASSETS set but ASSET_VENV empty"; exit 1; }; \
        source switch_env "${ASSET_VENV}" && \
        download_assets ${INSTALL_MIRROR_OPTION} --dir /opt/assets --assets "${ASSETS}" && \
        link_assets; \
    fi

# Consistency pass: some installs (e.g. gr00t --env isaaclab) pull pandas early, while
# numpy is still 2.x, then pin numpy back to 1.26. pandas >= 3.1 needs numpy >= 2.0.2,
# so re-pin pandas to the last series that supports numpy 1.26 in any venv with numpy < 2.
RUN for v in ${UV_PATH}/*/bin/python; do \
        [ -x "$v" ] || continue; \
        np=$("$v" -c 'import numpy; print(numpy.__version__)' 2>/dev/null) || continue; \
        pd=$("$v" -c 'import importlib.metadata as m; print(m.version("pandas"))' 2>/dev/null) || continue; \
        case "$np" in 1.*) \
            case "$pd" in 3.1*|3.[2-9]*) \
                echo "$(dirname $(dirname $v)): numpy $np + pandas $pd -> pinning pandas<3.1"; \
                uv pip install --python "$v" "pandas>=2.2,<3.1" "numpy==$np" || exit 1;; \
            esac;; \
        esac; \
        "$v" -c 'import pandas' >/dev/null 2>&1 || { echo "ERROR: pandas $pd still fails to import with numpy $np in $v"; exit 1; }; \
    done
# flash-attn for Blackwell RTX (sm_120, g7e RTX PRO 6000). The upstream prebuilt wheel
# (flash-attn 2.8.3 for torch 2.11) ships only sm_80 + sm_90 kernels. sm_89 (L40S)
# runs the sm_80 code, but sm_120 cannot: GR00T then fails at the first attention call
# with "no kernel image is available". When FLASH_ATTN_ARCHS is set (e.g. "8.0;9.0;12.0"),
# rebuild flash-attn from source for those archs in every venv that has it. Slow: about
# one compile pass per arch.
ARG FLASH_ATTN_ARCHS=""
# Parallel nvcc jobs. Each sm_80 backward kernel needs several GB of RAM; 16 jobs
# OOM-killed the build on CodeBuild BUILD_GENERAL1_2XLARGE (145 GB).
ARG FLASH_ATTN_MAX_JOBS=6
RUN if [ -n "${FLASH_ATTN_ARCHS}" ]; then \
      for v in ${UV_PATH}/*/bin/python; do \
        [ -x "$v" ] || continue; \
        fa=$("$v" -c 'import importlib.metadata as m; print(m.version("flash-attn").split("+")[0])' 2>/dev/null) || continue; \
        echo "=== $(dirname $(dirname $v)): rebuilding flash-attn ${fa} for ${FLASH_ATTN_ARCHS} ==="; \
        TORCH_CUDA_ARCH_LIST="${FLASH_ATTN_ARCHS}" FLASH_ATTN_CUDA_ARCHS="$(echo ${FLASH_ATTN_ARCHS} | tr -d '.')" \
        FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=${FLASH_ATTN_MAX_JOBS} NVCC_THREADS=2 \
          uv pip install --python "$v" --no-build-isolation --no-cache --reinstall-package flash-attn "flash-attn==${fa}" || exit 1; \
        "$v" -c 'import flash_attn, flash_attn_2_cuda; print("flash_attn", flash_attn.__version__, "rebuilt")' || exit 1; \
      done; \
    fi
# Isaac Sim standalone, baked into the image (opt-in). Set ISAAC_SIM_VERSION (e.g. 5.1.0)
# and ACCEPT_ISAAC_SIM_EULA=Y: by building with it you accept the NVIDIA Isaac Sim /
# Omniverse license (https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html).
# The image then runs without FSx staging: run_training_eks.sh uses /opt/isaacsim when it
# exists (ISAAC_PATH=auto or unset with ISAAC_SIM=1). The download stays in your own build (your private ECR); nothing is
# redistributed. Adds ~19 GB unpacked. Its own layer, so profile rebuilds reuse it.
ARG ISAAC_SIM_VERSION=""
ARG ACCEPT_ISAAC_SIM_EULA=""
RUN if [ -n "${ISAAC_SIM_VERSION}" ]; then \
      [ "${ACCEPT_ISAAC_SIM_EULA}" = Y ] || { echo "ERROR: set ACCEPT_ISAAC_SIM_EULA=Y to accept the NVIDIA Isaac Sim license"; exit 1; }; \
      url="https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-${ISAAC_SIM_VERSION}-linux-x86_64.zip"; \
      echo "=== Isaac Sim ${ISAAC_SIM_VERSION}: ${url} ==="; \
      (command -v unzip >/dev/null || { apt-get update -qq && apt-get install -y -qq unzip >/dev/null; }); \
      curl -fL --retry 5 --retry-delay 10 -o /tmp/isaacsim.zip "${url}" && \
      mkdir -p /opt/isaacsim && unzip -q -o /tmp/isaacsim.zip -d /opt/isaacsim && rm -f /tmp/isaacsim.zip && \
      test -f /opt/isaacsim/setup_conda_env.sh && \
      echo "Isaac Sim $(cat /opt/isaacsim/VERSION 2>/dev/null) -> /opt/isaacsim ($(du -sh /opt/isaacsim | cut -f1))"; \
    fi
# Mirror upstream final-image cleanup, and record which venvs this profile ships
# (read by the stage-2 DreamZero guard and the L2 container test).
RUN uv cache prune && rm -rf /var/lib/apt/lists/* && \
    for t in ${INSTALLS}; do echo "${t%%:*}"; done > "${UV_PATH}/.profile" && \
    echo "Profile venvs:" && cat "${UV_PATH}/.profile"

WORKDIR /workspace
CMD ["/bin/bash"]
