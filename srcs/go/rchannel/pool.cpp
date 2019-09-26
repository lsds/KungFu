#include <cstdlib>
#include <queue>
#include <stdexcept>
#include <unordered_map>

#include "pool.h"

struct pool_s {
  private:
    const size_t block_size_;
    std::unordered_map<void *, size_t> allocs_;
    std::unordered_map<size_t, std::queue<void *>> queues_;

    void check_leak() const
    {
        if (!allocs_.empty()) {
            throw std::runtime_error("memory leak detected.");
        }
    }

    void *alloc(size_t size)
    {
        void *ptr    = std::malloc(size);
        allocs_[ptr] = size;
        return ptr;
    }

    void free(void *ptr)
    {
        if (allocs_.count(ptr) == 0) {
            throw std::runtime_error("invalid free");
        }
        allocs_.erase(ptr);
        std::free(ptr);
    }

    static size_t align_to(size_t size, size_t block_size)
    {
        const size_t r = size % block_size;
        if (r != 0) { return size + block_size - r; }
        return size;
    }

    size_t align(size_t size) { return align_to(size, block_size_); }

  public:
    pool_s(size_t block_size = 512) : block_size_(block_size) {}

    ~pool_s() { check_leak(); }

    void *get(size_t size)
    {
        const size_t aligned = align(size);
        auto &q              = queues_[aligned];
        if (q.empty()) { return alloc(aligned); }
        void *ptr = q.front();
        q.pop();
        return ptr;
    }

    void put(void *ptr)
    {
        const auto it = allocs_.find(ptr);
        if (it == allocs_.end()) {
            throw std::invalid_argument("address not allocated");
        }
        const size_t aligned = it->second;
        queues_[aligned].push(ptr);
    }
};

pool_t *new_pool() { return new pool_s; }

void del_pool(pool_t *p) { delete p; }

void *get_buffer(pool_t *p, int size) { return p->get(size); }

void put_buffer(pool_t *p, void *ptr) { p->put(ptr); }
