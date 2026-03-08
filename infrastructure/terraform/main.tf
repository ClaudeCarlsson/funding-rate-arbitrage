terraform {
  required_providers {
    equinix = {
      source = "equinix/equinix"
    }
  }
}

provider "equinix" {
  auth_token = var.equinix_auth_token
}

variable "equinix_auth_token" {
  type      = string
  sensitive = true
}

variable "project_id" {
  type = string
}

# 1. The Brain Node (US-East / NY Area)
resource "equinix_metal_device" "brain_node" {
  hostname         = "arb-brain-us-east"
  plan             = "c3.medium.x86" # High-compute bare metal
  facilities       = ["ny5"]         # Close to major financial hubs
  operating_system = "ubuntu_22_04"
  billing_cycle    = "hourly"
  project_id       = var.project_id
}

# 2. Execution Node (Tokyo - Binance Proximity)
resource "equinix_metal_device" "exec_tokyo" {
  hostname         = "arb-exec-tokyo"
  plan             = "c3.small.x86"
  facilities       = ["ty11"]
  operating_system = "ubuntu_22_04"
  billing_cycle    = "hourly"
  project_id       = var.project_id
}

# 3. Execution Node (Singapore - Bybit Proximity)
resource "equinix_metal_device" "exec_singapore" {
  hostname         = "arb-exec-singapore"
  plan             = "c3.small.x86"
  facilities       = ["sg1"]
  operating_system = "ubuntu_22_04"
  billing_cycle    = "hourly"
  project_id       = var.project_id
}

output "brain_ip" {
  value = equinix_metal_device.brain_node.access_public_ipv4
}

output "tokyo_ip" {
  value = equinix_metal_device.exec_tokyo.access_public_ipv4
}

output "singapore_ip" {
  value = equinix_metal_device.exec_singapore.access_public_ipv4
}
