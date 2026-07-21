# Container Format Specification

## Overview

.omnia is a single-file container for medical imaging studies. It bundles all slices of a study into one file with a precomputed offset table for O(1) random access to any slice.

## Binary layout

```
┌──────────────────────────────────────────────┐
│ Header (16 bytes)                             │
│  • Magic: "OMN2" (4 bytes)                   │
│  • Version: uint8                            │
│  • Codec: uint8                              │
│  • Reserved: uint16                          │
│  • Metadata size: uint64                     │
├──────────────────────────────────────────────┤
│ Metadata (variable, Zstd-compressed JSON)     │
│  • num_slices                                │
│  • shape: [height, width]                    │
│  • dtype: numpy dtype string                 │
│  • chunk_sizes: [size_0, size_1, ...]        │
│  • crc32: [crc_0, crc_1, ...]               │
│  • original_bytes: total uncompressed size   │
├──────────────────────────────────────────────┤
│ Chunk 0 (variable)                           │
│  • CRC32: uint32 (of compressed chunk)       │
│  • Compressed slice data                     │
├──────────────────────────────────────────────┤
│ Chunk 1                                      │
│  • CRC32: uint32                             │
│  • Compressed slice data                     │
├──────────────────────────────────────────────┤
│ ...                                          │
├──────────────────────────────────────────────┤
│ Chunk N                                      │
│  • CRC32: uint32                             │
│  • Compressed slice data                     │
└──────────────────────────────────────────────┘
```

## Offset table

The offset table is computed from `chunk_sizes` at open time:

```
offsets[0] = 0
offsets[i] = sum(chunk_sizes[0:i]) for i > 0
```

Reading slice `i`:
1. Seek to `header_size + metadata_size + offsets[i]`
2. Read `chunk_sizes[i]` bytes
3. Verify CRC32
4. Decompress

Total time: O(1) seek + decompress. No other slices touched.

## Compression

Each slice is compressed independently using a fast lossless codec. Independent compression enables O(1) random access — decompressing slice 147 does not require decompressing slices 0–146.

## Integrity

Each chunk stores a CRC32 checksum of the compressed data. On read, the checksum is verified before decompression. Corruption is detected and reported immediately.

## Thread safety

Multiple threads can read from the same .omnia file concurrently. Each read operation is stateless — seek, read, verify, decompress. No shared mutable state.
