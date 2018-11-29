#pragma once
#include <array>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace std
{
namespace experimental
{
template <typename Ts, typename P, typename E, std::size_t... I>
static constexpr std::array<P, sizeof...(I)>
get_type_sizes(std::index_sequence<I...>)
{
    return {P{
        E::template value<typename std::tuple_element<I, Ts>::type>(),
        sizeof(typename std::tuple_element<I, Ts>::type),
    }...};
}

template <typename encoding> class basic_type_encoder
{
  public:
    using value_type = typename encoding::value_type;

    template <typename R> static constexpr value_type value()
    {
        return encoding::template value<R>();
    }

    static std::size_t size(const value_type type)
    {
        static constexpr int N =
            std::tuple_size<typename encoding::types>::value;
        using P = std::pair<value_type, std::size_t>;

        static constexpr std::array<P, N> type_sizes =
            get_type_sizes<typename encoding::types, P, encoding>(
                std::make_index_sequence<N>());

        for (int i = 0; i < N; ++i) {
            if (type_sizes[i].first == type) { return type_sizes[i].second; }
        }
        throw std::invalid_argument("invalid scalar code");
    }
};

}  // namespace experimental
}  // namespace std
