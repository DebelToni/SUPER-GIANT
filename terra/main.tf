terraform {
  required_version = ">= 1.7"
  required_providers {
    external = {
      source  = "hashicorp/external"
      version = "~> 2.3"
    }
  }
}

resource "null_resource" "runpod_gpu" {
  triggers = {
    pod_name          = var.pod_name
    image_name        = var.image_name
    gpu_type          = var.gpu_type
    container_disk_gb = var.container_disk_gb
    volume_gb         = var.volume_gb
  }

  provisioner "local-exec" {
    command = "${path.module}/scripts/runpod-cloud.sh create ${var.pod_name} ${var.image_name} \"${var.gpu_type}\" ${var.container_disk_gb} ${var.volume_gb}"
  }

  provisioner "local-exec" {
    when    = destroy
    command = "${path.module}/scripts/runpod-cloud.sh destroy GIANT-training"
  }
}

data "external" "runpod_ip" {
  program = [
    "${path.module}/scripts/runpod-cloud.sh",
    "ip",
    var.pod_name
  ]
  depends_on = [null_resource.runpod_gpu]
}

output "public_ip" {
  value = data.external.runpod_ip.result.ip
}
