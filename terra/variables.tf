variable "pod_name" {
  description = "Human-friendly name for the RunPod GPU instance"
  type        = string
  default     = "GIANT-training"
}

variable "gpu_type" {
  description = "RunPod gpuType string (see docs.runpod.io)"
  type        = string
  # default     = "NVIDIA RTX 3090" 
  # default     = "NVIDIA RTX 2000 Ada" 
  default     = "NVIDIA GeForce RTX 4090" 
}

variable "image_name" {
  description = "Container image to boot"
  type        = string
  default     = "docker.io/bananc/giant-training:latest"
}

variable "container_disk_gb" {
  type    = number
  default = 10
}
variable "volume_gb" {
  type    = number
  default = 20
}

variable "max_cost_usd_per_hour" {
  description = "Optional price ceiling for on-demand pods"
  type        = number
  default     = 0.5
}

