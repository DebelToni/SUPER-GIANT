/**
 * variables.tf – tweak these and `terraform apply` again
 */

variable "pod_name" {
  description = "Human-readable RunPod Pod name"
  type        = string
  default     = "tf-demo-ada"
}

variable "gpu_type" {
  description = <<-EOT
    Exact GPU string as listed in docs.runpod.io → GPU types.
    Must match the --gpuType flag for runpodctl.
  EOT
  type    = string
  default = "NVIDIA RTX 2000 Ada Generation" # 16 GB VRAM
}

variable "image_name" {
  description = "Container image that the pod will boot"
  type        = string
  default     = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-ubuntu22.04"
}

variable "container_disk_gb" {
  type        = number
  default     = 10
}

variable "volume_gb" {
  type        = number
  default     = 20
}

variable "price_ceiling" {
  description = "Optional $/hr ceiling. Leave null for 'cheapest available'."
  type        = number
  default     = null
}

