#define _GNU_SOURCE
#include "usrbio_bridge.c"

/* Independent research extension of gs_io. A source fragment maps to a
 * registered write slot; a destination fragment maps directly from a read
 * slot. All fragments and slots remain owned through the complete CQE drain.
 * This removes intermediate copies, not the mandatory registered-slot copy. */
static int vs_validate(struct bridge *b, int n, const uint64_t *sizes, int m,
                       const int *requests, const uint64_t *starts,
                       const uint64_t *lengths, void **ptrs, int read) {
    if (n < 1 || n > b->depth || m < 0) return -EINVAL;
    uint64_t covered[128] = {0};
    for (int i = 0; i < n; ++i)
        if (!sizes[i] || sizes[i] > b->slot_size) return -EINVAL;
    for (int j = 0; j < m; ++j) {
        int i = requests[j];
        if (i < 0 || i >= n || !ptrs[j] || !lengths[j] ||
            starts[j] > sizes[i] || lengths[j] > sizes[i] - starts[j])
            return -EINVAL;
        if (!read) {
            if (starts[j] != covered[i]) return -EINVAL;
            covered[i] += lengths[j];
        }
    }
    if (!read)
        for (int i = 0; i < n; ++i)
            if (covered[i] != sizes[i]) return -EINVAL;
    return 0;
}

static long vs_transfer(void *handle, int fd, int read, int n,
                        const uint64_t *offsets, const uint64_t *sizes, int m,
                        const int *requests, const uint64_t *starts,
                        const uint64_t *lengths, void **ptrs,
                        uint64_t *copied) {
    struct bridge *b = handle;
    int valid =
        vs_validate(b, n, sizes, m, requests, starts, lengths, ptrs, read);
    if (valid) return valid;
    *copied = 0;
    if (!read)
        for (int j = 0; j < m; ++j) {
            memcpy(b->iov.base + requests[j] * b->slot_size + starts[j],
                   ptrs[j], lengths[j]);
            *copied += lengths[j];
        }
    struct hf3fs_ior *ring = read ? &b->reader : &b->writer;
    int prepared = 0, error = 0;
    for (int i = 0; i < n; ++i) {
        int r =
            hf3fs_prep_io(ring, &b->iov, read, b->iov.base + i * b->slot_size,
                          fd, offsets[i], sizes[i], (void *)(uintptr_t)(i + 1));
        if (r < 0) {
            error = r;
            break;
        }
        ++prepared;
    }
    int submitted = hf3fs_submit_ios(ring);
    if (submitted < 0) error = submitted;
    unsigned char seen[128] = {0};
    int done = 0;
    while (done < prepared) {
        struct hf3fs_cqe cqes[128];
        int count = hf3fs_wait_for_ios(ring, cqes, prepared - done, 1, NULL);
        if (count < 0)
            abort(); /* Unknown completion cannot release ownership. */
        for (int j = 0; j < count; ++j) {
            uintptr_t id = (uintptr_t)cqes[j].userdata;
            if (!id || id > (uintptr_t)prepared || seen[id - 1]) abort();
            int i = (int)id - 1;
            seen[i] = 1;
            if (cqes[j].result != (int64_t)sizes[i])
                error = cqes[j].result < 0 ? (int)cqes[j].result : -EIO;
            ++done;
        }
    }
    /* A short/error batch publishes no scatter output, including successful
     * peers. Every submitted buffer has nevertheless finished before return. */
    if (error) return error;
    if (read)
        for (int j = 0; j < m; ++j) {
            memcpy(ptrs[j],
                   b->iov.base + requests[j] * b->slot_size + starts[j],
                   lengths[j]);
            *copied += lengths[j];
        }
    long bytes = 0;
    for (int i = 0; i < n; ++i) bytes += (long)sizes[i];
    return bytes;
}

long vs_write(void *handle, int fd, int n, const uint64_t *offsets,
              const uint64_t *sizes, int m, const int *requests,
              const uint64_t *starts, const uint64_t *lengths, void **ptrs,
              uint64_t *copied) {
    return vs_transfer(handle, fd, 0, n, offsets, sizes, m, requests, starts,
                       lengths, ptrs, copied);
}

long vs_read(void *handle, int fd, int n, const uint64_t *offsets,
             const uint64_t *sizes, int m, const int *requests,
             const uint64_t *starts, const uint64_t *lengths, void **ptrs,
             uint64_t *copied) {
    return vs_transfer(handle, fd, 1, n, offsets, sizes, m, requests, starts,
                       lengths, ptrs, copied);
}

uint64_t vs_copy(int n, void **sources, void **destinations,
                 const uint64_t *lengths) {
    uint64_t copied = 0;
    for (int i = 0; i < n; ++i) {
        memcpy(destinations[i], sources[i], lengths[i]);
        copied += lengths[i];
    }
    return copied;
}
