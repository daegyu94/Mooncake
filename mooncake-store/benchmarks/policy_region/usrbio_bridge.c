#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <hf3fs_usrbio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* One owner per ring. Slots stay alive until every CQE has been drained.
 * No timeout/cancellation release: outstanding buffers must not be reused. */
struct bridge {
    struct hf3fs_iov iov;
    struct hf3fs_ior reader, writer;
    size_t slot_size;
    int depth;
};
void *gs_create(const char *mount, size_t slot_size, int depth) {
    if (!slot_size || depth < 1 || depth > 128) return NULL;
    struct bridge *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->slot_size = slot_size;
    b->depth = depth;
    if (hf3fs_iovcreate(&b->iov, mount, slot_size * depth, 0, -1)) goto fail;
    if (hf3fs_iorcreate(&b->reader, mount, depth, true, 0, -1)) goto iov;
    if (hf3fs_iorcreate(&b->writer, mount, depth, false, 0, -1)) goto reader;
    return b;
reader:
    hf3fs_iordestroy(&b->reader);
iov:
    hf3fs_iovdestroy(&b->iov);
fail:
    free(b);
    return NULL;
}
void gs_destroy(void *handle) {
    struct bridge *b = handle;
    hf3fs_iordestroy(&b->writer);
    hf3fs_iordestroy(&b->reader);
    hf3fs_iovdestroy(&b->iov);
    free(b);
}
int gs_open(const char *path) {
    int fd = open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) return -errno;
    int r = hf3fs_reg_fd(fd, 0);
    if (r > 0) {
        close(fd);
        return -r;
    }
    return fd;
}
void gs_close(int fd) {
    hf3fs_dereg_fd(fd);
    close(fd);
}
/* return bytes or negative errno; userdata identifies slots, not CQE order. */
long gs_io(void *handle, int fd, int read, int n, const uint64_t *offsets,
           const uint64_t *sizes, void **buffers) {
    struct bridge *b = handle;
    if (n < 1 || n > b->depth) return -EINVAL;
    for (int i = 0; i < n; ++i)
        if (!sizes[i] || sizes[i] > b->slot_size || !buffers[i]) return -EINVAL;
    struct hf3fs_ior *ring = read ? &b->reader : &b->writer;
    int prepared = 0, error = 0;
    for (int i = 0; i < n; ++i) {
        void *slot = b->iov.base + i * b->slot_size;
        if (!read) memcpy(slot, buffers[i], sizes[i]);
        int r = hf3fs_prep_io(ring, &b->iov, read, slot, fd, offsets[i],
                              sizes[i], (void *)(uintptr_t)(i + 1));
        if (r < 0) {
            error = r;
            break;
        }
        ++prepared;
    }
    int submitted = hf3fs_submit_ios(ring);
    /* Even on submit failure, drain prepared IO before releasing slots. */
    if (submitted < 0) error = submitted;
    unsigned char seen[128] = {0};
    int done = 0;
    while (done < prepared) {
        struct hf3fs_cqe cqes[128];
        int count = hf3fs_wait_for_ios(ring, cqes, prepared - done, 1, NULL);
        if (count <
            0) { /* Cannot prove completion; fail closed, no slot reuse. */
            abort();
        }
        for (int j = 0; j < count; ++j) {
            uintptr_t id = (uintptr_t)cqes[j].userdata;
            if (!id || id > (uintptr_t)prepared || seen[id - 1]) abort();
            int i = id - 1;
            seen[i] = 1;
            if (cqes[j].result != (int64_t)sizes[i])
                error = cqes[j].result < 0 ? cqes[j].result : -EIO;
            else if (read)
                memcpy(buffers[i], b->iov.base + i * b->slot_size, sizes[i]);
            ++done;
        }
    }
    if (error) return error;
    long bytes = 0;
    for (int i = 0; i < n; ++i) bytes += sizes[i];
    return bytes;
}
