locals {
  # Kubernetes namespace / service account the workloads run as (see k8s/base.yaml).
  k8s_namespace       = "accelbench"
  k8s_service_account = "accelbench-runner"

  accelerator_pool_defaults = {
    disk_size_gb = 200
    disk_type    = "pd-balanced"
  }
}

resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "tpu.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# --- Network ------------------------------------------------------------------

resource "google_compute_network" "vpc" {
  name                    = "${var.name}-vpc"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "${var.name}-subnet"
  region        = var.region
  network       = google_compute_network.vpc.id
  ip_cidr_range = "10.10.0.0/20"

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.20.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.30.0.0/20"
  }
}

# --- Storage & registry -------------------------------------------------------

resource "google_storage_bucket" "results" {
  name                        = "${var.project_id}-${var.name}-results"
  location                    = var.region
  uniform_bucket_level_access = true
  # Lets `terraform destroy` remove the bucket. Run `make fetch-results` first.
  force_destroy = true
  depends_on    = [google_project_service.apis]
}

resource "google_artifact_registry_repository" "images" {
  repository_id = var.name
  location      = var.region
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# --- IAM ----------------------------------------------------------------------

# Least-privilege identity for the nodes themselves (logging, metrics, image pulls).
resource "google_service_account" "nodes" {
  account_id   = "${var.name}-nodes"
  display_name = "AccelBench GKE nodes"
}

resource "google_project_iam_member" "nodes" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/artifactregistry.reader",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.nodes.email}"
}

# Identity the pods use (via Workload Identity) to read/write the results bucket.
resource "google_service_account" "workload" {
  account_id   = "${var.name}-workload"
  display_name = "AccelBench workloads"
}

resource "google_storage_bucket_iam_member" "workload_results" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.workload.email}"
}

resource "google_service_account_iam_member" "workload_identity" {
  service_account_id = google_service_account.workload.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${local.k8s_namespace}/${local.k8s_service_account}]"
  depends_on         = [google_container_cluster.this]
}

# --- GKE cluster --------------------------------------------------------------

resource "google_container_cluster" "this" {
  name     = var.name
  location = var.zone

  network    = google_compute_network.vpc.id
  subnetwork = google_compute_subnetwork.subnet.id

  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false

  release_channel {
    channel = "REGULAR"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  addons_config {
    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }

  depends_on = [google_project_service.apis]
}

# Small always-on pool for system pods, the benchmark client and CPU-only jobs.
resource "google_container_node_pool" "system" {
  name       = "system"
  cluster    = google_container_cluster.this.id
  location   = var.zone
  node_count = 1

  node_config {
    machine_type    = "e2-standard-4"
    disk_size_gb    = 100
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

resource "google_container_node_pool" "gpu" {
  name               = "gpu"
  cluster            = google_container_cluster.this.id
  location           = var.zone
  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = var.gpu_max_nodes
  }

  node_config {
    machine_type    = var.gpu_machine_type
    spot            = var.use_spot
    disk_size_gb    = local.accelerator_pool_defaults.disk_size_gb
    disk_type       = local.accelerator_pool_defaults.disk_type
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels          = { "accelbench/pool" = "gpu" }

    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count

      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

resource "google_container_node_pool" "tpu_train" {
  name               = "tpu-train"
  cluster            = google_container_cluster.this.id
  location           = var.zone
  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = var.tpu_max_nodes
  }

  node_config {
    machine_type    = var.tpu_train_machine_type
    spot            = var.use_spot
    disk_size_gb    = local.accelerator_pool_defaults.disk_size_gb
    disk_type       = local.accelerator_pool_defaults.disk_type
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels          = { "accelbench/pool" = "tpu-train" }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

resource "google_container_node_pool" "tpu_serve" {
  name               = "tpu-serve"
  cluster            = google_container_cluster.this.id
  location           = var.zone
  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = var.tpu_max_nodes
  }

  node_config {
    machine_type    = var.tpu_serve_machine_type
    spot            = var.use_spot
    disk_size_gb    = local.accelerator_pool_defaults.disk_size_gb
    disk_type       = local.accelerator_pool_defaults.disk_type
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels          = { "accelbench/pool" = "tpu-serve" }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}
