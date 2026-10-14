#pragma once

#include <optional>

namespace mooncake {

struct TransferSubmitterConfig {
    // Unset leaves transport-dependent automatic selection to the submitter.
    std::optional<bool> memcpy_enabled_override;
    bool parallel_dfs_put = false;

    static TransferSubmitterConfig FromEnvironment();
};

}  // namespace mooncake
