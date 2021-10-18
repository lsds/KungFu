#include <algorithm>
#include <stdml/bits/data/state2.hpp>
#include <stdml/bits/vfs/vfs.hpp>
#include <stdml/elastic_state.hpp>

namespace stdml::data
{
void mount_tf_index(vfs::tree &root, const data::total_index &index);
void mount_tf_record_shards(vfs::tree &root, const data::total_index &index,
                            const int64_t progress, const int cluster_size,
                            const int rank, state2 ds, int global_batch_size,
                            size_t max_sample_per_file);

void mount_tf_record_for_progress(vfs::tree &root,
                                  const data::total_index &index,
                                  const int64_t progress,
                                  const int cluster_size, state2 ds,
                                  int global_batch_size,
                                  size_t max_sample_per_file);
}  // namespace stdml::data
