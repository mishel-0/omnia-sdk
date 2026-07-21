# .omnia Container Format Specification (Training Edition)

## Binary Layout

```
Offset  Size  Field
─────────────────────────────────────
0       4     Magic: "OMN2" (0x324E4D4F)
4       1     Format version (3 = Training Edition)
5       1     Codec: 0 = Zstd, 1 = JPEG 2000
6       2     Reserved (zero)
8       8     Metadata section size (uint64 LE)
16      M     Metadata (Zstd-compressed JSON)
16+M    ...   Pixel chunks (Zstd-compressed, concatenated)
```

## Metadata JSON Schema

```json
{
  "v": 3,              // Format version
  "codec": "zstd",     // Codec name
  "n": 1108,           // Number of slices
  "sz": [241234, ...], // Compressed size of each slice chunk
  "dtype": "int16",    // NumPy dtype string
  "shape": [512, 512], // Slice dimensions (H, W)
  "original_bytes": 580911104,  // Total uncompressed pixel bytes
  "s": [{"f": "slice_0001.dcm"}, ...]  // Optional per-slice metadata
}
```

## Reading Algorithm

1. Read 4 bytes, verify magic == `b"OMN2"`
2. Read version, codec, reserved (4 bytes)
3. Read 8 bytes metadata size (uint64 LE)
4. Read and Zstd-decompress metadata
5. For slice `i`: seek to `16 + M + sum(sz[0:i])`, read `sz[i]` bytes, Zstd-decompress
6. Convert raw bytes to numpy array using `dtype` and `shape`

## Codec IDs

| ID | Codec | Use |
|----|-------|-----|
| 0 | Zstd | Training edition (fast decode) |
| 1 | JPEG 2000 | PACS edition (higher compression) |
