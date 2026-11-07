region       = "us-east-2"
cluster_name = "rlinf-on-eks"
# RT-core simulation training (examples/isaaclab-gr00t-ppo). AZ of the system node group,
# and the default AZ for FSx and the GPU NodePools. Empty = the region's first AZ.
# Do not change it to chase GPU capacity (that replaces the system node group): move FSx
# with TF_VAR_fsx_az (storage, then workloads; see "Scarce GPU capacity" in infrastructure/AGENTS.md).
target_az = ""
