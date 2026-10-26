# =============================================================================
# RLinf Container Image for Amazon EKS
#
# Strategy: Two-stage build.
#   Stage 1: Build upstream RLinf image using their Dockerfile + BUILD_TARGET.
#   Stage 2: Layer EFA networking stack on top for multi-node NCCL on EKS.
#
# This preserves upstream's dependency management (uv, multi-venv, install.sh)
# while adding the AWS EFA stack that upstream doesn't include.
#
# The BUILD_TARGET arg selects which stage-1 image to layer on (tag
# rlinf-upstream-${BUILD_TARGET}):
#   - embodied-maniskill_libero  (ManiSkill + LIBERO, 13 model venvs; `full` profile)
#   - profile-<name>             (slim image from examples/Dockerfile.profile)
#   - embodied-robotwin          (RoboTwin, 3 model venvs)
#   - embodied-calvin            (CALVIN, 2 model venvs)
#   - embodied-metaworld         (MetaWorld, 2 model venvs)
#   - embodied-behavior-openvlaoft (BEHAVIOR, 1 model venv)
#   - reason                     (Agentic RL, vLLM/SGLang)
#   - ... (20 targets total, see upstream docker/Dockerfile)
#
# Build (from rlinf-on-eks repo root):
#   docker build \
#     --build-arg BUILD_TARGET=embodied-maniskill_libero \
#     -t rlinf-eks:maniskill_libero \
#     -f examples/Dockerfile .
#
# The build context MUST be this repo's root (COPY paths reference examples/).
# The .dockerignore at repo root keeps the context small.
# =============================================================================

# ---- Stage 1: Build upstream RLinf image ----
ARG BUILD_TARGET=embodied-maniskill_libero
ARG UPSTREAM_DOCKERFILE=docker/Dockerfile

FROM rlinf-upstream-${BUILD_TARGET} AS upstream
# This FROM is a placeholder -- the actual upstream build happens in CodeBuild
# (see buildspec.yml). CodeBuild builds the upstream image first, tags it as
# rlinf-upstream-${BUILD_TARGET}, then this Dockerfile layers EFA on top.
# For local builds, run the upstream build first:
#   cd /path/to/RLinf
#   docker build --build-arg BUILD_TARGET=embodied-maniskill_libero \
#     -t rlinf-upstream-embodied-maniskill_libero -f docker/Dockerfile .


# ---- Stage 2: EFA networking overlay ----
# Layer EFA, GDRCopy, NCCL, and OpenMPI onto the upstream RLinf image.
# Recipe proven in nccl-tests/Dockerfile on nvidia/cuda base images.

FROM rlinf-upstream-${BUILD_TARGET}

ARG GDRCOPY_VERSION=v2.5.1
ARG EFA_INSTALLER_VERSION=1.50.0
ARG NCCL_VERSION=v2.31.2-1
# nccl-tests v2.21.0+ needs NCCL >= 2.32 (NCCL_WIN_GIN_ONLY); v2.20.0 builds against 2.31.2.
ARG NCCL_TESTS_VERSION=v2.20.0

ENV DEBIAN_FRONTEND=noninteractive

# -- Clean slate: remove any pre-existing RDMA/NCCL packages --
# The upstream CUDA base may ship old ibverbs/NCCL that conflict with EFA.
RUN apt-get update -y && \
    apt-get remove -y --allow-change-held-packages \
        ibverbs-utils libibverbs-dev libibverbs1 libmlx5-1 \
        libnccl2 libnccl-dev 2>/dev/null || true && \
    rm -rf /opt/hpcx /usr/local/mpi && \
    rm -f /etc/ld.so.conf.d/hpcx.conf && \
    ldconfig
ENV OPAL_PREFIX=

# -- System dependencies for EFA/NCCL build --
# NOTE: The EFA installer requires libevent-pthreads, libhwloc15, udev,
# environment-modules, and tcl which the upstream RLinf base may not include.
# Install them explicitly to avoid "held broken packages" errors.
RUN apt-get install -y --no-install-recommends \
    autoconf automake build-essential cmake curl \
    environment-modules \
    git gcc gdb kmod \
    libevent-pthreads-2.1-7 libhwloc15 \
    libtool pkg-config \
    openssh-client openssh-server \
    tcl udev \
    && rm -rf /var/lib/apt/lists/*

# Purge cuda-compat to avoid NCCL crashes when the image's compat libs do not
# match the host driver. Still required on the CUDA 12.8.1 base (main@6f46f36a+).
RUN apt-get purge -y cuda-compat-* 2>/dev/null || true

# -- SSH config for MPI (required for MPIJob multi-node) --
RUN mkdir -p /var/run/sshd && \
    sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# -- Library paths (set early so subsequent installs are found) --
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:/opt/gdrcopy/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/opt/gdrcopy/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}
ENV LIBRARY_PATH=/opt/gdrcopy/lib:${LIBRARY_PATH:-}
ENV CPATH=/opt/gdrcopy/include:${CPATH:-}

# =============================================================================
# GDRCopy -- GPU Direct RDMA copy library (userspace)
#
# Required for GPU-initiated DMA transfers over EFA. The kernel module
# (gdrdrv) must be loaded on the host node; we only build userspace here.
# Documented prerequisite: AGENTS.md Discovery #43.
# =============================================================================
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy && \
    cd /tmp/gdrcopy && \
    make prefix=/opt/gdrcopy install && \
    rm -rf /tmp/gdrcopy

# =============================================================================
# EFA Installer 1.50.0
#
# CRITICAL: Must use 1.47.0+ (not 1.34.0/1.38.0 which bundle old libfabric 1.22).
# Installer 1.50.0 bundles (https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-changelog.html):
#   - libfabric 2.6.0amzn1.0 (RDMA + DMA-BUF support)
#   - aws-ofi-nccl 1.21.1 (installed to /opt/amazon/ofi-nccl)
#   - EFA driver 3.3.0 (kmod skipped here: --skip-kmod; the host provides it)
#   - rdma-core 64.0
#   - Open MPI 4.1.7 (/opt/amazon/openmpi -- the one on PATH/OPAL_PREFIX) and
#     Open MPI 5.0.9 (/opt/amazon/openmpi5)
# 1.50.0 drops Amazon Linux 2 support; irrelevant here (Ubuntu 22.04 image).
#
# No --minimal flag: we need the full install including aws-ofi-nccl.
# --disable-ngc is REQUIRED: since 1.48.0 (not in 1.47.0) the installer auto-detects an
# "NGC container" via /opt/nvidia/nvidia_entrypoint.sh, which the nvidia/cuda base
# ships. In NGC mode it installs libnccl-ofi-ngc-v3 instead of libnccl-ofi, which
# lacks libnccl-ofi-tuner.so (NCCL_TUNER_PLUGIN below) and the libnccl-net.so
# symlink. -g is a documented no-op (GDR is on by default); -d installs debug pkgs.
# The libevent-pthreads and libhwloc15 deps are installed above.
# See AGENTS.md Discoveries #41-#45.
# =============================================================================
RUN cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify --disable-ngc && \
    rm -rf /tmp/aws-efa-installer*

# Verify aws-ofi-nccl was installed by the EFA installer
RUN echo "Verifying AWS OFI NCCL plugin installation..." && \
    (ls -la /opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-ofi*.so || \
     ls -la /opt/amazon/ofi-nccl/lib/aarch64-linux-gnu/libnccl-ofi*.so || \
     ls -la /opt/amazon/ofi-nccl/lib/libnccl-ofi*.so)

# =============================================================================
# NCCL from source
#
# Build NCCL for target GPU architectures.
# sm_80 (A100), sm_86 (A10G), sm_89 (L4/L40S), sm_90 (H100/H200), sm_100 (B200)
# Upstream RLinf base is CUDA 12.8.1 (since main@6f46f36a), which supports sm_100.
# NCCL v2.31.2-1 is the version aws-ofi-nccl 1.21.1 (EFA installer 1.50.0) is
# tested with (https://github.com/aws/aws-ofi-nccl/releases/tag/v1.21.1).
# Known issue: NCCL 2.31.2 PAT+NVLS regresses on H100. PAT is off by default; do
# not set NCCL_ALGO=PAT.
# =============================================================================
RUN git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git /opt/nccl && \
    cd /opt/nccl && \
    make -j $(nproc) src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 \
                  -gencode=arch=compute_86,code=sm_86 \
                  -gencode=arch=compute_89,code=sm_89 \
                  -gencode=arch=compute_90,code=sm_90 \
                  -gencode=arch=compute_100,code=sm_100"

# nccl-tests built against OUR NCCL + Amazon OpenMPI, so L3 measures this image's
# exact NCCL/aws-ofi-nccl/EFA stack (not a third-party image).
# MPI_HOME=/opt/amazon/openmpi (Open MPI 4.1.7) matches PATH and OPAL_PREFIX.
RUN git clone -b ${NCCL_TESTS_VERSION} --depth 1 https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests && \
    cd /opt/nccl-tests && \
    make -j $(nproc) MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/opt/nccl/build \
    NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89"

# -- OpenMPI tuning --
ENV OMPI_MCA_pml=^ucx \
    OMPI_MCA_btl=tcp,self \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent \
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth \
    PMIX_MCA_gds=hash

# Preload source-built NCCL over any system NCCL
ENV LD_PRELOAD=/opt/nccl/build/lib/libnccl.so

# -- EFA runtime defaults --
ENV FI_EFA_USE_HUGE_PAGE=0
ENV NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so

# -- EKS-specific environment --
# MuJoCo / ManiSkill headless rendering (osmesa, not egl -- Discovery #8)
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# NVIDIA driver capabilities -- required for ManiSkill GPU rendering
ENV NVIDIA_DRIVER_CAPABILITIES=all

# PyTorch / training defaults
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TOKENIZERS_PARALLELISM=true
ENV TORCH_NCCL_AVOID_RECORD_STREAMS=1
ENV NCCL_DEBUG=WARN

# -- Copy EKS-specific scripts --
COPY examples/scripts/ /workspace/eks/scripts/

# =============================================================================
# Optional extras (controlled by EXTRAS build arg)
#
# EXTRAS is a comma-separated list of packages to install on top of the EFA
# overlay. This allows the same Dockerfile to produce images for different
# examples without rebuilding the expensive EFA/NCCL layers.
#
# Supported values:
#   rlinf       - RLinf source (required for all training examples)
#
# DreamZero note: the DreamZero `groot` package is provided by the
# COPY DreamZero/ tree on PYTHONPATH (DREAMZERO_PATH). Its `dreamzero` venv ships
# in upstream's embodied-maniskill_libero target (the `full` profile); the guard
# below fails a full build that lacks it. Profile builds (examples/Dockerfile.profile)
# write ${UV_PATH}/.profile and only need dreamzero if they list it.
#
# Examples:
#   --build-arg EXTRAS=rlinf                  (ManiSkill, LIBERO, pi0, DreamZero)
# =============================================================================
ARG EXTRAS="rlinf"

# Copy source repos into build context.
# RLinf is always available (cloned in pre_build).
# DreamZero (groot package) is always cloned by the buildspec (pre_build) and
# made available on PYTHONPATH via DREAMZERO_PATH; it is no longer gated by EXTRAS.
COPY RLinf/ /workspace/RLinf/
COPY DreamZero/ /workspace/DreamZero/

# DreamZero runs in the `dreamzero` venv. Full image (no .profile): upstream's
# embodied-maniskill_libero target ships it, so require it. Profile image: require
# it only if the profile lists it.
RUN if [ -f "${UV_PATH}/.profile" ]; then \
        echo "Profile image; venvs: $(tr '\n' ' ' < "${UV_PATH}/.profile")"; \
        if grep -qx dreamzero "${UV_PATH}/.profile" && [ ! -d "${UV_PATH}/dreamzero" ]; then \
            echo "ERROR: profile lists dreamzero but ${UV_PATH}/dreamzero is missing"; exit 1; fi; \
    else \
        test -d "${UV_PATH}/dreamzero" || { echo "ERROR: BUILD_TARGET lacks the dreamzero venv; use embodied-maniskill_libero or a profile"; exit 1; }; \
    fi

# Wan world-model example deps overlay.
# The wan-openvlaoft-grpo example runs in the `openvla-oft` venv (switch_env
# openvla-oft) but needs deps the base openvla-oft venv lacks. Upstream's
# install_wan_world_model() does two things: (1) editable-install the Wan
# diffusion library `diffsynth` from github.com/RLinf/diffsynth-studio (the
# Wan env imports `diffsynth` at load), then (2) install
# requirements/embodied/models/wan.txt. venvs are isolated, so we add both to
# the openvla-oft venv only. Idempotent.
# numpy is pinned to the venv's existing version: these deps are unpinned, and
# pandas>=3.1 (2026-10-22) requires numpy>=2, which pip would install over the
# venv's numpy 1.26.x and break tensorflow 2.15 / ml_dtypes
# (`numpy.core.umath failed to import`) for ManiSkill PPO and Wan GRPO.
RUN if [ -d "${UV_PATH}/openvla-oft" ]; then \
        echo "Installing Wan world-model deps into openvla-oft venv..."; \
        np_ver=$("${UV_PATH}/openvla-oft/bin/python" -c 'import numpy; print(numpy.__version__)') && \
        echo "numpy==${np_ver}" > /tmp/wan-constraints.txt && \
        echo "Constraining Wan deps to numpy==${np_ver}" && \
        git clone --depth 1 https://github.com/RLinf/diffsynth-studio.git /opt/diffsynth-studio && \
        "${UV_PATH}/openvla-oft/bin/pip" install --no-cache-dir -c /tmp/wan-constraints.txt -e /opt/diffsynth-studio && \
        "${UV_PATH}/openvla-oft/bin/pip" install --no-cache-dir -c /tmp/wan-constraints.txt \
            safetensors sentencepiece protobuf modelscope ftfy pandas \
            'deepspeed==0.18.4' seaborn && \
        test "$("${UV_PATH}/openvla-oft/bin/python" -c 'import numpy; print(numpy.__version__)')" = "${np_ver}" && \
        rm -f /tmp/wan-constraints.txt; \
    else \
        echo "openvla-oft venv absent; skipping Wan deps overlay."; \
    fi

# Install extras based on EXTRAS arg
RUN chmod +x /workspace/eks/scripts/install_extras.sh && \
    /workspace/eks/scripts/install_extras.sh "${EXTRAS}"

# =============================================================================
# Security hardening / image hygiene (runs last so it cleans everything above)
#
# 1. Apply available OS security upgrades for fixable HIGH-severity CVEs
#    (gnupg/gpg family CVE-2025-68973, linux-libc-dev kernel headers).
# 2. Purge build-time caches (uv/pip/git checkouts under /opt/venv/.cache and
#    /root/.cache). These are NOT needed at runtime and account for the bulk of
#    third-party CVE/misconfig findings (vendored Go/Rust binaries and cached
#    upstream Dockerfiles from transitive dependencies).
# 3. Remove bundled Dockerfiles shipped inside installed dependency source trees
#    (gr00t, dexbotic, RLinf, nsight). They are upstream artifacts, never built
#    here, and only trigger IaC-misconfig findings. NOTE: matched case-sensitively
#    (`-name`, not `-iname`) and with site-packages/dist-packages EXCLUDED, so we
#    never delete Python modules literally named `dockerfile.py` (e.g.
#    wandb/sdk/launch/builder/templates/dockerfile.py) -- a case-insensitive match
#    previously corrupted wandb and crashed EnvWorker init in the venvs.
#
# Trivy evidence: this removes the fixable CRITICAL (gRPC in wandb-core),
# ~36 cache vulnerabilities, and all 55 HIGH Dockerfile misconfigurations.
# =============================================================================
RUN apt-get update -y && \
    apt-get install -y --only-upgrade --no-install-recommends \
        dirmngr gnupg gnupg-l10n gnupg-utils gnupg2 \
        gpg gpg-agent gpg-wks-client gpg-wks-server gpgconf gpgsm gpgv \
        openssl libssl3 \
        linux-libc-dev 2>/dev/null || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /opt/venv/.cache /root/.cache /root/.cargo/registry /tmp/* && \
    find /opt/venv -path '*/.cache' -type d -prune -exec rm -rf {} + 2>/dev/null || true && \
    find /opt /workspace /usr -type f \
        -not -path '*/site-packages/*' -not -path '*/dist-packages/*' \
        \( -name 'Dockerfile' -o -name 'Dockerfile.*' -o -name '*.Dockerfile' \) \
        -delete 2>/dev/null || true && \
    ldconfig

# Upgrade build/packaging utilities (and their setuptools-vendored copies) that
# carry fixable HIGH CVEs (jaraco.context CVE-2026-23949, wheel CVE-2026-24049).
# setuptools is capped `<81`: 81 removed the bundled `pkg_resources`, which
# `sapien` (ManiSkill's physics backend) imports at load time -- an unbounded
# upgrade pulls setuptools 83 and breaks `import mani_skill` with
# `ModuleNotFoundError: No module named 'pkg_resources'`. `>=78.1.1` still
# satisfies the CVE fix while `<81` keeps pkg_resources available.
# These are used only at package-build time. Run across every model venv.
# A final cache purge runs LAST here: pip/uv re-create /opt/venv/.cache during the
# upgrade above (after the hardening block's purge), and that regenerated cache
# carries third-party files that trip the scanner (e.g. a JWT-shaped string in
# scikit-image's data fetcher). Purging at the very end keeps the image clean.
ENV PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1
RUN for v in /opt/venv/*/bin/activate; do \
        [ -f "$v" ] || continue; \
        venv_dir="$(dirname "$(dirname "$v")")"; \
        echo "Upgrading packaging tools in ${venv_dir}"; \
        set +u; . "$v"; set -u; \
        pip install --no-cache-dir --upgrade 'setuptools>=78.1.1,<81' 'wheel>=0.46.2' 2>/dev/null || \
            echo "  WARNING: tooling upgrade skipped in ${venv_dir}"; \
        deactivate 2>/dev/null || true; \
    done && \
    command -v uv >/dev/null 2>&1 && uv cache clean 2>/dev/null || true; \
    rm -rf /opt/venv/.cache /root/.cache /tmp/* && \
    find /opt/venv -path '*/.cache' -type d -prune -exec rm -rf {} + 2>/dev/null || true

WORKDIR /workspace/RLinf

# DreamZero groot package is provided via PYTHONPATH at runtime (see launchers).
ENV DREAMZERO_PATH=/workspace/DreamZero

CMD ["/bin/bash"]
