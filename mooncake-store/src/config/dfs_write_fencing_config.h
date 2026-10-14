#pragma once

#include "environ.h"
#include "environment_variables.h"

namespace mooncake {
struct DfsWriteFencingConfig {
    static bool Enabled() {
        return Environ::ReadOr(
            TransferSubmitterEnvironmentVariables::MC_STORE_DFS_WRITE_FENCING,
            false);
    }
};
inline constexpr const char* kFencedDfsEpochFile =
    ".mooncake_fenced_writer_epoch";
}  // namespace mooncake
