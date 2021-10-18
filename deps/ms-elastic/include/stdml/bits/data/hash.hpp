#pragma once
#include <ttl/device>
#include <ttl/nn/bits/kernels/hash.hpp>
#include <ttl/nn/ops>
#include <ttl/tensor>
#include <type_traits>

namespace stdml::data
{
template <typename N>
class crc
{
    static_assert(std::is_unsigned<N>::value, "");

    N table[256];

    N sum(const uint8_t *s, const uint8_t *t,
          const N init = static_cast<N>(-1)) const
    {
        return std::accumulate(s, t, init, this->operator());
    }

    N t(const N s, const N poly)
    {
        return (s & 1) == 1 ? (s >> 1) ^ poly : s >> 1;
    }

  public:
    explicit crc(const N poly)
    {
        for (uint32_t i = 0; i < 256; i++) {
            N crc = i;
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            crc = t(crc, poly);
            table[i] = crc;
        }
    }

    N operator()(N a, uint32_t b) const
    {
        return (a >> 8) ^ table[static_cast<uint8_t>(a ^ b)];
    }

    template <typename R, ttl::internal::rank_t r>
    void operator()(const ttl::tensor_ref<N, 0> &y,
                    const ttl::tensor_view<R, r> &x) const
    {
        y.data()[0] = sum(reinterpret_cast<const uint8_t *>(x.data()),
                          reinterpret_cast<const uint8_t *>(x.data_end())) ^
                      static_cast<N>(-1);
    }
};

struct crc_polynomials {
    // static constexpr uint16_t usb = static_cast<uint16_t>(0xa001);
    static constexpr uint32_t ieee = static_cast<uint32_t>(0xedb88320);
    // static constexpr uint64_t ecma =
    // static_cast<uint64_t>(0xC96C5795D7870F42);
};

// using crc16_usb = standard_crc<uint16_t, crc_polynomials::usb>;
// using crc32_ieee = standard_crc<uint32_t, crc_polynomials::ieee>;
// using crc64_ecma = standard_crc<uint64_t, crc_polynomials::ecma>;

class hasher
{
    using N = uint32_t;
    using crc_t = crc<N>;

    N sum_;
    const crc_t crc_;

  public:
    hasher(N poly = crc_polynomials::ieee)
        : sum_(static_cast<N>(-1)), crc_(poly)
    {
    }

    void sync(const std::function<N(N)> &f)
    {
        sum_ = f(sum_);
    }

    void operator()(uint8_t a)
    {
        sum_ = crc_(sum_, a);
    }

    template <typename T>
    void operator()(const T &a)
    {
        const uint8_t *p = reinterpret_cast<const uint8_t *>(&a);
        const uint8_t *q = p + sizeof(T);
        for (; p < q; ++p) {
            (*this)(*p);
        }
    }

    operator N() const
    {
        return sum_ ^ static_cast<N>(-1);
    }
};
}  // namespace stdml::data
