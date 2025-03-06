.. _disk_storage:

Dealing With Disks
==================
During the process of training and serving a LLM, you could easily run out of
disk space on your VM or container. Potential large storage objects may
include:

* Base model (e.g. from Hugging Face)
* Finetuning Dataset
* Output checkpoints

This page presents a few potential options for provisioning additional
disk space for your deployments.


General Recommendations
-----------------------
We generally recommend the following best practices:

* Download your model or dataset to a shared storage volume or bucket
  first, and mount this volume or bucket when you need it. This would
  save both disk space and time spent on downloading large files.
* Prefer writing checkpoints from a single worker. This avoids having
  to manage multiple writes to the same shared drive.


Persistent Disks
----------------
Persistent Disks (PDs) are durable storage volumes that can be
dynamically provisioned at deployment time. PDs provide block storage,
which is similar to having a hard drive attached to your VM. These
are a versatile solution and are ideal for multi-read, low latency
workloads.

* Provisioning on TPU VMs: https://cloud.google.com/tpu/docs/setup-persistent-disk#:~:text=Attach%20a%20Persistent%20Disk%20when%20you%20create%20a%20TPU%20VM&text=To%20determine%20which%20VM%20image,you%20create%20a%20TPU%20VM.

* Provisioning on GKE: https://cloud.google.com/kubernetes-engine/docs/concepts/persistent-volumes


Google Cloud Storage Buckets
----------------------------
GCS buckets are an alternative solution to PDs. Unlike PDs, GCS buckets
provide object storage. These are ideal for storing large, unstructured
data such as datasets and model checkpoints.

Kithara is designed to work seamlessly with GCS buckets. To use a GCS
bucket, make sure that the service account which is used to run your
workload has write access to the GCS bucket.

.. code-block:: bash

  gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET_NAME \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT_EMAIL" \
    --role="YOUR_IAM_ROLE"


The IAM role should be either 'roles/storage.objectCreator' or 'roles/storage.objectAdmin'.


GCS Fuse
--------
An alternative way to use GCS Buckets is to mount them as local drives,
with GCS Fuse. After mounting the bucket, you can access the files in the
bucket as if they were in a locally mounted volume. GCSFuse buckets can
be read-only or read-write. You could even mount the same bucket on multiple
workers.

* GCE VM: https://cloud.google.com/storage/docs/cloud-storage-fuse/overview

* GKE: https://cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-setup#enable

  * Sample Kubernetes yaml: https://github.com/AI-Hypercomputer/kithara/blob/main/ray/TPU/GKE/single-host.yaml

  * Note - running on GKE requires enabling workload identity first: https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity



