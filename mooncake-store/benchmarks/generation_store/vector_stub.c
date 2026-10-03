/* Test-only POSIX provider for the HF3FS C ABI. Never a 3FS benchmark. */
#define _GNU_SOURCE
#include <errno.h>
#include <hf3fs_usrbio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct pending {
    void *slot;
    int fd;
    size_t off;
    uint64_t size;
    const void *userdata;
    int64_t result;
};
struct stub_ring {
    struct pending items[128];
    int count;
    bool read;
};
static int short_index = -1, prep_error_index = -1, submit_error = 0;
static int wait_error = 0, drained = 0, prepared_count = 0;
void vs_stub_config(int short_slot, int prep_slot, int submit, int wait) {
    short_index = short_slot;
    prep_error_index = prep_slot;
    submit_error = submit;
    wait_error = wait;
    drained = prepared_count = 0;
}
int vs_stub_drained(void) { return drained; }
int vs_stub_prepared(void) { return prepared_count; }
int hf3fs_iovcreate(struct hf3fs_iov *iov, const char *mount, size_t size,
                    size_t block_size, int numa) {
    (void)mount;
    (void)block_size;
    (void)numa;
    iov->base = malloc(size);
    iov->size = size;
    return iov->base ? 0 : -ENOMEM;
}
void hf3fs_iovdestroy(struct hf3fs_iov *iov) { free(iov->base); }
int hf3fs_iorcreate(struct hf3fs_ior *ior, const char *mount, int entries,
                    bool read, int depth, int numa) {
    (void)mount;
    (void)entries;
    (void)depth;
    (void)numa;
    struct stub_ring *ring = calloc(1, sizeof(*ring));
    if (!ring) return -ENOMEM;
    ring->read = read;
    ior->iorh = ring;
    return 0;
}
void hf3fs_iordestroy(struct hf3fs_ior *ior) { free(ior->iorh); }
int hf3fs_reg_fd(int fd, uint64_t flags) {
    (void)fd;
    (void)flags;
    return 0;
}
void hf3fs_dereg_fd(int fd) { (void)fd; }
int hf3fs_prep_io(const struct hf3fs_ior *ior, const struct hf3fs_iov *iov,
                  bool read, void *ptr, int fd, size_t off, uint64_t len,
                  const void *userdata) {
    (void)iov;
    (void)read;
    struct stub_ring *ring = ior->iorh;
    int index = ring->count;
    if (index == prep_error_index) return -EIO;
    ring->items[ring->count++] =
        (struct pending){ptr, fd, off, len, userdata, 0};
    ++prepared_count;
    return index;
}
int hf3fs_submit_ios(const struct hf3fs_ior *ior) {
    struct stub_ring *ring = ior->iorh;
    for (int i = 0; i < ring->count; ++i) {
        struct pending *p = &ring->items[i];
        size_t len = p->size - (i == short_index ? 1 : 0);
        p->result = ring->read ? pread(p->fd, p->slot, len, p->off)
                               : pwrite(p->fd, p->slot, len, p->off);
    }
    return submit_error ? -EIO : 0;
}
int hf3fs_wait_for_ios(const struct hf3fs_ior *ior, struct hf3fs_cqe *cqes,
                       int count, int min_results,
                       const struct timespec *timeout) {
    (void)count;
    (void)min_results;
    (void)timeout;
    if (wait_error) return -EIO;
    struct stub_ring *ring = ior->iorh;
    if (!ring->count) return 0;
    /* Reverse ordering and one completion per wait exercise stable userdata. */
    struct pending *p = &ring->items[--ring->count];
    cqes[0] = (struct hf3fs_cqe){0, 0, p->result, p->userdata};
    ++drained;
    return 1;
}
