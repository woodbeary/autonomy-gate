#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fd_path(int fd, char *out, size_t size) {
    char link[64];
    snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(link, out, size - 1);
    if (n < 0) return -1;
    out[n] = 0;
    return 0;
}

static void copy_fd(int fd, const char *durable, const char *name) {
    char dst[PATH_MAX], tmp[PATH_MAX];
    snprintf(dst, sizeof dst, "%s/%s", durable, name);
    snprintf(tmp, sizeof tmp, "%s/.%s.partial", durable, name);
    int out = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) return;
    char buf[65536];
    off_t off = 0;
    ssize_t n;
    while ((n = pread(fd, buf, sizeof buf, off)) > 0) {
        if (write(out, buf, (size_t)n) != n) break;
        off += n;
    }
    close(out);
    rename(tmp, dst);
}

static void save_listing(const char *dir, const char *durable) {
    char dst[PATH_MAX], tmp[PATH_MAX];
    snprintf(dst, sizeof dst, "%s/.listing", durable);
    snprintf(tmp, sizeof tmp, "%s/.listing.partial", durable);
    FILE *out = fopen(tmp, "w");
    if (!out) return;
    DIR *d = opendir(dir);
    if (d) {
        struct dirent *e;
        while ((e = readdir(d)) != NULL) {
            if (e->d_name[0] == '.') continue;
            fprintf(out, "%s\n", e->d_name);
        }
        closedir(d);
    }
    fclose(out);
    rename(tmp, dst);
}

static void record(int fd) {
    const char *dir = getenv("AG_POWERLOSS_DIR");
    const char *durable = getenv("AG_POWERLOSS_DURABLE");
    char path[PATH_MAX];
    if (!dir || !durable || fd_path(fd, path, sizeof path)) return;
    if (strcmp(path, dir) == 0) {
        save_listing(dir, durable);
        return;
    }
    size_t len = strlen(dir);
    if (strncmp(path, dir, len) != 0 || path[len] != '/') return;
    if (strstr(path, " (deleted)")) return;
    copy_fd(fd, durable, path + len + 1);
}

int fsync(int fd) {
    static int (*real)(int);
    if (!real) real = (int (*)(int))dlsym(RTLD_NEXT, "fsync");
    int rc = real(fd);
    if (rc == 0) record(fd);
    return rc;
}

int fdatasync(int fd) {
    static int (*real)(int);
    if (!real) real = (int (*)(int))dlsym(RTLD_NEXT, "fdatasync");
    int rc = real(fd);
    if (rc == 0) record(fd);
    return rc;
}
