region       = "us-east-1" # pick a region that offers g7e or g6e on-demand (or an ODCR)
cluster_name = "rlinf-on-eks"
# RT-core simulation training (examples/isaaclab-gr00t-ppo). Cluster layer only: the gpu-sim
# NodePool itself is enabled in the workloads layer (TF_VAR_enable_sim_pool=true). AZ of the system node group,
# and the default AZ for FSx and the GPU NodePools. Empty = the region's first AZ.
# Do not change it to chase GPU capacity (that replaces the system node group): move FSx
# with TF_VAR_fsx_az (storage, then workloads; see "Scarce GPU capacity" in infrastructure/AGENTS.md).
target_az = ""
