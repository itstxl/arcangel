#include "gguf_file.h"

#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ink {

const TypeInfo * type_info(uint32_t type) {
    // block sizes from ggml/src/ggml-common.h
    static const std::map<uint32_t, TypeInfo> table = {
        {0, {"F32", 1, 4}},        {1, {"F16", 1, 2}},        {30, {"BF16", 1, 2}},
        {8, {"Q8_0", 32, 34}},     {12, {"Q4_K", 256, 144}},  {13, {"Q5_K", 256, 176}},
        {14, {"Q6_K", 256, 210}},  {16, {"IQ2_XXS", 256, 66}}, {17, {"IQ2_XS", 256, 74}},
        {18, {"IQ3_XXS", 256, 98}}, {19, {"IQ1_S", 256, 50}},  {22, {"IQ2_S", 256, 82}},
        {23, {"IQ4_XS", 256, 136}}, {39, {"MXFP4", 32, 17}},  {40, {"NVFP4", 64, 36}},
        {42, {"Q2_0", 64, 18}},
    };
    auto it = table.find(type);
    return it == table.end() ? nullptr : &it->second;
}

int64_t GgufValue::as_int() const {
    switch (kind) {
        case U64: return (int64_t) u;
        case I64: return i;
        case F64: return (int64_t) f;
        case BOOL: return b;
        default: throw std::runtime_error("gguf: value is not numeric");
    }
}

double GgufValue::as_float() const {
    switch (kind) {
        case U64: return (double) u;
        case I64: return (double) i;
        case F64: return f;
        case BOOL: return b;
        default: throw std::runtime_error("gguf: value is not numeric");
    }
}

namespace {

struct Cursor {
    const uint8_t * p;
    const uint8_t * end;

    void need(size_t n) const {
        if ((size_t) (end - p) < n) throw std::runtime_error("gguf: truncated file");
    }
    template <typename T> T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = read<uint64_t>();
        need(n);
        std::string s((const char *) p, n);
        p += n;
        return s;
    }
};

GgufValue read_value(Cursor & c, uint32_t t) {
    GgufValue v;
    switch (t) {
        case 0: v.kind = GgufValue::U64; v.u = c.read<uint8_t>(); break;
        case 1: v.kind = GgufValue::I64; v.i = c.read<int8_t>(); break;
        case 2: v.kind = GgufValue::U64; v.u = c.read<uint16_t>(); break;
        case 3: v.kind = GgufValue::I64; v.i = c.read<int16_t>(); break;
        case 4: v.kind = GgufValue::U64; v.u = c.read<uint32_t>(); break;
        case 5: v.kind = GgufValue::I64; v.i = c.read<int32_t>(); break;
        case 6: v.kind = GgufValue::F64; v.f = c.read<float>(); break;
        case 7: v.kind = GgufValue::BOOL; v.b = c.read<uint8_t>() != 0; break;
        case 8: v.kind = GgufValue::STRING; v.s = c.str(); break;
        case 9: {
            v.kind = GgufValue::ARRAY;
            const uint32_t et = c.read<uint32_t>();
            const uint64_t n = c.read<uint64_t>();
            v.arr.reserve(n);
            for (uint64_t k = 0; k < n; ++k) v.arr.push_back(read_value(c, et));
            break;
        }
        case 10: v.kind = GgufValue::U64; v.u = c.read<uint64_t>(); break;
        case 11: v.kind = GgufValue::I64; v.i = c.read<int64_t>(); break;
        case 12: v.kind = GgufValue::F64; v.f = c.read<double>(); break;
        default: throw std::runtime_error("gguf: unknown value type " + std::to_string(t));
    }
    return v;
}

}  // namespace

GgufFile::GgufFile(const std::string & path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st {};
    if (fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("gguf: cannot stat " + path); }
    size_ = (uint64_t) st.st_size;
    map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map_ == MAP_FAILED) { map_ = nullptr; ::close(fd); throw std::runtime_error("gguf: mmap failed for " + path); }
    fd_ = fd;

    const uint8_t * base = (const uint8_t *) map_;
    Cursor c{base, base + size_};
    if (c.read<uint32_t>() != 0x46554747u) throw std::runtime_error("gguf: bad magic");  // "GGUF"
    const uint32_t version = c.read<uint32_t>();
    if (version != 3) throw std::runtime_error("gguf: unsupported version " + std::to_string(version));
    const uint64_t n_tensors = c.read<uint64_t>();
    const uint64_t n_kv = c.read<uint64_t>();

    for (uint64_t k = 0; k < n_kv; ++k) {
        std::string key = c.str();
        const uint32_t t = c.read<uint32_t>();
        meta_[key] = read_value(c, t);
    }

    tensors_.reserve(n_tensors);
    for (uint64_t k = 0; k < n_tensors; ++k) {
        GgufTensor t;
        t.name = c.str();
        t.n_dims = c.read<uint32_t>();
        if (t.n_dims > 4) throw std::runtime_error("gguf: tensor " + t.name + " has more than 4 dims");
        for (uint32_t d = 0; d < t.n_dims; ++d) t.ne[d] = (int64_t) c.read<uint64_t>();
        t.type = c.read<uint32_t>();
        t.offset = c.read<uint64_t>();
        const TypeInfo * ti = type_info(t.type);
        if (!ti) throw std::runtime_error("gguf: tensor " + t.name + " has unsupported type " + std::to_string(t.type));
        if (t.ne[0] % ti->block_elems) throw std::runtime_error("gguf: tensor " + t.name + " row is not a whole number of blocks");
        t.n_bytes = (uint64_t) (t.n_elements() / ti->block_elems) * ti->block_bytes;
        by_name_[t.name] = tensors_.size();
        tensors_.push_back(std::move(t));
    }

    uint64_t align = 32;
    if (auto it = meta_.find("general.alignment"); it != meta_.end()) align = (uint64_t) it->second.as_int();
    const uint64_t header = (uint64_t) (c.p - base);
    const uint64_t data_start = (header + align - 1) / align * align;
    for (auto & t : tensors_) {
        if (data_start + t.offset + t.n_bytes > size_) throw std::runtime_error("gguf: tensor " + t.name + " runs past the end of the file");
        t.data = base + data_start + t.offset;
    }
}

GgufFile::~GgufFile() {
    if (map_) munmap(map_, size_);
    if (fd_ >= 0) ::close(fd_);
}

void GgufFile::release(const uint8_t * p, size_t n) const {
    if (!map_ || n == 0) return;
    const uint8_t * base = (const uint8_t *) map_;
    const size_t pg = 4096, off = (size_t) (p - base);
    const size_t a = (off + pg - 1) / pg * pg, b = (off + n) / pg * pg;  // whole pages inside the range only
    if (b <= a) return;
    madvise((void *) (base + a), b - a, MADV_DONTNEED);
    if (fd_ >= 0) posix_fadvise(fd_, (off_t) a, (off_t) (b - a), POSIX_FADV_DONTNEED);
}

const GgufValue * GgufFile::find(const std::string & key) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? nullptr : &it->second;
}

const GgufValue & GgufFile::get(const std::string & key) const {
    const GgufValue * v = find(key);
    if (!v) throw std::runtime_error("gguf: missing key " + key);
    return *v;
}

const GgufTensor * GgufFile::tensor(const std::string & name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

}  // namespace ink
