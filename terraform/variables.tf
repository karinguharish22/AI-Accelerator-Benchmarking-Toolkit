variable "project_id" {
  description = "Google Cloud project ID."
  type        = string
}

variable "region" {
  description = "Region for the bucket and Artifact Registry."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone for the (zonal) GKE cluster. Must offer both the GPU type and TPU v5e."
  type        = string
  default     = "us-central1-a"
}

variable "name" {
  description = "Prefix used for every resource name."
  type        = string
  default     = "accelbench"
}

variable "use_spot" {
  description = "Run accelerator node pools on Spot VMs (much cheaper, can be preempted)."
  type        = bool
  default     = true
}

# --- GPU pool -----------------------------------------------------------------

variable "gpu_machine_type" {
  description = "Machine type for the GPU node pool (g2-standard-8 = 1x NVIDIA L4)."
  type        = string
  default     = "g2-standard-8"
}

variable "gpu_type" {
  description = "GPU accelerator type attached to each GPU node."
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_count" {
  description = "GPUs per node."
  type        = number
  default     = 1
}

variable "gpu_max_nodes" {
  description = "Autoscaling ceiling for the GPU pool. The floor is always 0."
  type        = number
  default     = 1
}

# --- TPU pools ----------------------------------------------------------------

variable "tpu_train_machine_type" {
  description = "Single-host TPU v5e machine for training (ct5lp-hightpu-4t = 4 chips, 2x2)."
  type        = string
  default     = "ct5lp-hightpu-4t"
}

variable "tpu_serve_machine_type" {
  description = "Single-host TPU v5e machine for serving (ct5lp-hightpu-1t = 1 chip, 1x1)."
  type        = string
  default     = "ct5lp-hightpu-1t"
}

variable "tpu_max_nodes" {
  description = "Autoscaling ceiling for each TPU pool. The floor is always 0."
  type        = number
  default     = 1
}
