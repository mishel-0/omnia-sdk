#ifndef OMNIA_CORE_H
#define OMNIA_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────── */
#define OMNIA_MAGIC       "OMI2"
#define OMNIA_VERSION     2
#define OMNIA_MAX_SLICES  4096
#define OMNIA_MAX_SIZE    (8UL * 1024 * 1024 * 1024)  /* 8 GB */

/* ─── Compression modes ──────────────────────────────────── */
typedef enum {
    OMNIA_COMPRESS_LOSSLESS_JP2K  = 0,  /* JPEG 2000 lossless (default) */
    OMNIA_COMPRESS_LOSSLESS_ZSTD  = 1,  /* ZSTD fallback */
    OMNIA_COMPRESS_LOSSY_HIGH     = 2,  /* JPEG 2000 lossy, high quality */
} omnia_compress_mode_t;

/* ─── Result type ────────────────────────────────────────── */
typedef struct {
    int      code;         /* 0 = success, <0 = error */
    char     message[256]; /* human-readable error */
    size_t   compressed_size;
    size_t   original_size;
    double   ratio;
} omnia_result_t;

/* ─── Slice metadata ─────────────────────────────────────── */
typedef struct {
    uint16_t rows;
    uint16_t cols;
    uint16_t bits_allocated;
    int16_t  window_center;
    uint16_t window_width;
    char     patient_id[64];
    char     study_uid[64];
    char     series_uid[64];
    char     sop_uid[64];
} omnia_slice_meta_t;

/* ─── Core API ───────────────────────────────────────────── */

/**
 * Compress a directory of DICOM slices to a single .omnia file.
 * @param input_dir   Path to directory containing .dcm files
 * @param output_path Path for the output .omnia file
 * @param mode        Compression mode (use 0 for lossless JP2K)
 * @return omnia_result_t with status
 */
omnia_result_t omnia_compress(const char* input_dir,
                              const char* output_path,
                              omnia_compress_mode_t mode);

/**
 * Decompress a .omnia file back to DICOM slices.
 * @param input_path  Path to .omnia file
 * @param output_dir  Path to directory for restored .dcm files
 * @return omnia_result_t with status
 */
omnia_result_t omnia_decompress(const char* input_path,
                                const char* output_dir);

/**
 * Get metadata from a .omnia file without decompressing.
 * @param input_path  Path to .omnia file
 * @param slices_out  Output array of slice metadata (caller frees)
 * @param count_out   Number of slices
 * @return omnia_result_t
 */
omnia_result_t omnia_info(const char* input_path,
                          omnia_slice_meta_t** slices_out,
                          size_t* count_out);

/**
 * Verify a .omnia file integrity (checksum check).
 * @param input_path  Path to .omnia file
 * @return omnia_result_t with code=0 if valid
 */
omnia_result_t omnia_verify(const char* input_path);

/**
 * Get version string.
 */
const char* omnia_version(void);

/**
 * Free memory allocated by omnia_info.
 */
void omnia_free_slices(omnia_slice_meta_t* slices, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* OMNIA_CORE_H */
