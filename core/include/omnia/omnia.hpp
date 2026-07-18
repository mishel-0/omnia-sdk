#ifndef OMNIA_CPP_H
#define OMNIA_CPP_H

#include <string>
#include <vector>
#include <optional>
#include <stdexcept>
#include "omnia/omnia.h"

namespace omnia {

/* ─── Result ─────────────────────────────────────────── */
struct Result {
    bool     success;
    std::string message;
    size_t   compressed_size;
    size_t   original_size;
    double   ratio;
};

/* ─── SliceInfo ───────────────────────────────────────── */
struct SliceInfo {
    uint16_t rows, cols;
    std::string patient_id;
    std::string study_uid;
};

/* ─── Compressor ──────────────────────────────────────── */
class Compressor {
public:
    explicit Compressor(omnia_compress_mode_t mode = OMNIA_COMPRESS_LOSSLESS_JP2K);
    
    Result compress(const std::string& input_dir,
                    const std::string& output_path);
    
    Result decompress(const std::string& input_path,
                      const std::string& output_dir);
    
    std::vector<SliceInfo> info(const std::string& input_path);
    
    bool verify(const std::string& input_path);
    
    static std::string version();

private:
    omnia_compress_mode_t mode_;
};

}  // namespace omnia
#endif
