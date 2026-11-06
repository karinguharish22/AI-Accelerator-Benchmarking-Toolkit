output "cluster_name" {
  value = google_container_cluster.this.name
}

output "zone" {
  value = var.zone
}

output "bucket" {
  value = google_storage_bucket.results.name
}

output "registry" {
  description = "Prefix for container images, e.g. <registry>/train-gpu:latest."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "workload_service_account" {
  description = "Google service account the Kubernetes service account impersonates."
  value       = google_service_account.workload.email
}

output "get_credentials" {
  value = "gcloud container clusters get-credentials ${google_container_cluster.this.name} --zone ${var.zone} --project ${var.project_id}"
}
