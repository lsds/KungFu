#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <stdml/bits/collective/connection.hpp>
#include <stdml/bits/collective/elastic.hpp>
#include <stdml/bits/collective/execution.hpp>
#include <stdml/bits/collective/http.hpp>
#include <stdml/bits/collective/json.hpp>
#include <stdml/bits/collective/log.hpp>

#include "libkungfu-elastic-cgo.h"

namespace stdml::collective
{
extern std::string safe_getenv(const char *name);

std::optional<cluster_config> go_get_cluster_config()
{
    std::vector<std::byte> buf(1 << 16);
    int len = GoReadConfigServer(buf.data(), (int)buf.size());
    if (len <= 0) {
        return {};
    }
    buf.resize(len);
    return cluster_config::from(buf).value();
}
}  // namespace stdml::collective
