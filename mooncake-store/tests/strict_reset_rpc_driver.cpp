#include <iostream>
#include <string>
#include <vector>
#include "master_client.h"

// Native metadata integration helper: hold a real RPC PutStart open until
// the controller releases this participant, then finish before reset.
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    using namespace mooncake;
    MasterClient client(generate_uuid());
    if (client.Connect(argv[1]) != ErrorCode::OK) return 3;
    std::vector<char> memory(16 * 1024 * 1024);
    Segment segment;
    segment.id = generate_uuid();
    segment.name = argv[2];
    segment.te_endpoint = segment.name;
    segment.base = reinterpret_cast<uintptr_t>(memory.data());
    segment.size = memory.size();
    if (!client.MountSegment(segment)) return 4;
    ReplicateConfig config;
    config.replica_num = 1;
    config.preferred_segment = segment.name;
    if (!client.PutStart(argv[2], {1024}, config)) return 5;
    std::cout << "READY" << std::endl;
    std::string command;
    if (!std::getline(std::cin, command) || command != "FINISH") return 6;
    if (!client.PutEnd(ObjectMeta{argv[2], std::nullopt}, ReplicaType::MEMORY))
        return 7;
    std::cout << "DRAINED" << std::endl;
    if (!std::getline(std::cin, command) || command != "CLOSE") return 8;
    if (!client.UnmountSegment(segment.id)) return 9;
    return 0;
}
