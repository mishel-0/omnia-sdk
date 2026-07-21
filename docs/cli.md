# CLI Reference

## Commands

### compress

```bash
omnia compress <input_dir> <output_dir>
```

Compress a directory of DICOM studies into .omnia files.

| Argument | Description |
|----------|-------------|
| `input_dir` | Directory containing DICOM study folders |
| `output_dir` | Output directory for .omnia files |

### extract

```bash
omnia extract <input.omnia> <output_dir>
```

Restore a .omnia file to DICOM files.

| Argument | Description |
|----------|-------------|
| `input.omnia` | .omnia file to extract |
| `output_dir` | Directory for restored DICOM files |

### verify

```bash
omnia verify <input.omnia>
```

Verify all slices in a .omnia file. Checks CRC32 on every chunk and reports any corruption.

### benchmark

```bash
omnia benchmark <dicom_dir> <omnia_dir>
```

Run a training benchmark comparing DICOM vs .omnia performance. Requires both raw DICOM and compressed .omnia versions of the same dataset.
