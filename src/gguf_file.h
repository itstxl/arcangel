// Read-only GGUF v3 reader. The file is memory-mapped PROT_READ: the engine never writes to disk.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ink {

// ggml type ids used by the Inkling files (ggml/include/ggml.h)
enum class GgmlType : uint32_t {
    F32 = 0, F16 = 1, Q5_0 = 6, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ2_XXS = 16, IQ2_XS = 17,
    IQ3_XXS = 18, IQ1_S = 19, IQ2_S = 22, IQ4_XS = 23, BF16 = 30, MXFP4 = 39, NVFP4 = 40, Q2_0 = 42,
};

struct TypeInfo {
    const char * name;
    uint32_t block_elems;  // elements per block
    uint32_t block_bytes;  // bytes per block
};

// nullptr for a type the engine does not know
const TypeInfo * type_info(uint32_t type);

struct GgufValue {
    enum Kind { U64, I64, F64, BOOL, STRING, ARRAY } kind = U64;
    uint64_t u = 0;
    int64_t i = 0;
    double f = 0;
    bool b = false;
    std::string s;
    std::vector<GgufValue> arr;

    int64_t as_int() const;   // any numeric kind
    double as_float() const;  // any numeric kind
};

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    int64_t ne[4] = {1, 1, 1, 1};  // ggml order: ne[0] is the contiguous dimension
    uint32_t n_dims = 0;
    uint64_t offset = 0;           // from the start of the data section
    uint64_t n_bytes = 0;
    const uint8_t * data = nullptr;  // into the mapping

    int64_t n_elements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};

class GgufFile {
public:
    explicit GgufFile(const std::string & path);  // throws std::runtime_error
    ~GgufFile();
    GgufFile(const GgufFile &) = delete;
    GgufFile & operator=(const GgufFile &) = delete;

    const std::map<std::string, GgufValue> & meta() const { return meta_; }
    const std::vector<GgufTensor> & tensors() const { return tensors_; }

    const GgufValue * find(const std::string & key) const;
    const GgufValue & get(const std::string & key) const;  // throws when missing
    const GgufTensor * tensor(const std::string & name) const;

    uint64_t file_size() const { return size_; }
    // Drop [p, p + n) of the mapping from this process and from the page cache once it has been copied: a 10-18 GB
    // model file otherwise fills the cache, and with a high vm.swappiness the kernel then swaps out other memory
    // (including offloaded experts) rather than dropping it.
    void release(const uint8_t * p, size_t n) const;

private:
    void * map_ = nullptr;
    int fd_ = -1;
    uint64_t size_ = 0;
    std::map<std::string, GgufValue> meta_;
    std::vector<GgufTensor> tensors_;
    std::map<std::string, size_t> by_name_;
};

}  // namespace ink
