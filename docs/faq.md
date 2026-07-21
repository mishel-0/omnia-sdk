# FAQ

### Why not ZIP?
ZIP requires full decompression to access a single slice. For a 277-slice CT study, reading slice 200 means decompressing slices 0–199 first. No random access.

### Why not HDF5?
HDF5 has no DICOM ingestion path. Concurrency model is fragile under high worker counts. Metadata overhead is significant for large studies.

### Why not Zarr?
Zarr has no DICOM understanding — no modality tags, no study hierarchy, no patient metadata. Converting DICOM to Zarr discards medical context.

### Why not NIfTI?
NIfTI was designed for neuroimaging with minimal metadata. No DICOM tag preservation. Gzip requires full file decompression.

### Why not WebDataset?
WebDataset shards tar files but still requires DICOM parsing per sample. No random access within a shard — reading one slice requires scanning the tar.

### Why not JPEG 2000?
JPEG 2000 is a per-slice codec, not a container. 16 ms decode per slice is too slow for training. .omnia's codec is 0.3 ms.

### Why not MONAI CacheDataset?
Caches in RAM — works for datasets under 10 GB but does not scale to multi-TB archives. No storage savings.

### Why not NVIDIA DALI?
DALI has no DICOM decoder. Requires custom operators. No storage savings.

### Why not AV1 / H.265?
Lossy — unacceptable for diagnostic CT where 1 HU differences matter. Random access is frame-granular, not slice-granular.

### Why not LMDB / SQLite?
Adds database complexity and licensing cost without providing compression or DICOM-native organization.

### Why not keep raw DICOM?
277 files per study causes 69M syscalls per epoch and 48% GPU utilization. .omnia keeps your existing PACS — the container is a cache format alongside it, not a replacement.
