terraform {
  required_version = ">= 1.10"
  required_providers {
    external = {
      source  = "hashicorp/external"
      version = "~> 2.3"
    }
  }
}

variable "pod_name" { default = "tf-demo" }

# Create on every apply (timestamp() forces replacement)
resource "null_resource" "runpod_create" {
  triggers = { always = timestamp() }

  provisioner "local-exec" {
    command = "${path.module}/scripts/runpod-cloud.sh create ${var.pod_name}"
  }

  provisioner "local-exec" {
    when    = destroy
    command = "${path.module}/scripts/runpod-cloud.sh destroy ${var.pod_name}"
  }
}

# Read data back
data "external" "runpod_ip" {
  depends_on = [null_resource.runpod_create]
  program = [
    "${path.module}/scripts/runpod-cloud.sh",
    "ip",
    "${var.pod_name}"
  ]
}

output "public_ip" {
  value = data.external.runpod_ip.result.ip
}

