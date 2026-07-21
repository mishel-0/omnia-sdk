# Dataset: LIDC-IDRI

The benchmark uses the **LIDC-IDRI** (Lung Image Database Consortium) dataset from The Cancer Imaging Archive (TCIA).

## Access

1. Visit https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
2. Download the DICOM files (requires NBIA Data Retriever or TCIA download tools)
3. Extract to a local directory

## Expected structure

```
lidc_raw/
├── LIDC-IDRI-0001/
│   ├── lidc_idri/
│   │   └── LIDC-IDRI-0001/
│   │       └── 1.3.6.1.4.1.14519.5.2.1.6279.6001.*/
│   │           ├── CT_1.3.6.1.4.1.14519.5.2.1.6279.6001.*/
│   │           │   ├── *.dcm  (CT slice files)
│   │           │   └── ...
│   │           ├── SEG_*/
│   │           └── SR_*/
│   └── ...
├── LIDC-IDRI-0002/
└── ...
```

## Reference benchmark subset

The reference benchmark uses 15 patients with complete CT series (3,387 slices total, 512×512 int16, uncompressed TransferSyntax).

## Preprocessing

No preprocessing is required. The `DicomDataset` class automatically filters to CT modality and 512×512 slices.
