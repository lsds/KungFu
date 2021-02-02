#include <hwloc.h>

#include <functional>
#include <vector>

namespace kungfu
{
class hw_top
{
    hwloc_topology_t topology;

    void
    traverse_numa(const hwloc_obj_t node,
                  const std::function<void(const hwloc_obj_t, const int)> &f,
                  const int depth) const
    {
        f(node, depth);
        for (int i = 0; i < static_cast<int>(node->arity); i++) {
            traverse_numa(node->children[i], f, depth + 1);
        }
    }

    std::vector<int> get_order(const char *type_name) const
    {
        std::vector<int> order;
        traverse_numa(
            hwloc_get_root_obj(topology),
            [&](const hwloc_obj_t node, const int) {
                char type[32];
                hwloc_obj_type_snprintf(type, sizeof(type), node, 0);
                if (strcmp(type_name, type) == 0) {
                    order.push_back(node->os_index);
                }
            },
            0);
        return order;
    }

  public:
    hw_top()
    {
        hwloc_topology_init(&topology);
        hwloc_topology_load(topology);
    }

    ~hw_top() { hwloc_topology_destroy(topology); }

    std::vector<int> get_pu_numa_order() const { return get_order("PU"); }

    // FIXME: may return 0 when numa node is 1
    size_t get_numa_node_count() const
    {
        const auto nodes = get_order("NUMANode");
        return nodes.size();
    }
};

static hw_top ht;

std::vector<int> get_pu_numa_order() { return ht.get_pu_numa_order(); }

size_t get_numa_node_count() { return ht.get_numa_node_count(); }
}  // namespace kungfu
