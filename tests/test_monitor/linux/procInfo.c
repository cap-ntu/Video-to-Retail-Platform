// Desc: Linux c program to get pid, name and resource usage of current process.
// Author: Zhou Shengsheng
// Date: 22/04/19
//
// References:
// Get system memory usage: https://www.cnblogs.com/emrysche/articles/9354363.html
// Get memory usage of current process: https://stackoverflow.com/questions/1558402/memory-usage-of-current-process-in-c
// Get cpu cores: https://www.cnblogs.com/airduce/p/9103945.html
// Get pid: http://man7.org/linux/man-pages/man2/getpid.2.html
// Get process name: https://stackoverflow.com/questions/9097201/how-to-get-current-process-name-in-linux
// Get proc info: http://man7.org/linux/man-pages/man2/getrusage.2.html

#include <stdio.h>
// For getrusage
#include <sys/time.h>
#include <sys/resource.h>
// For random test
#include <time.h>
#include <stdlib.h>
// For getpid
#include <sys/types.h>
#include <unistd.h>

#define _GNU_SOURCE
#include <errno.h>

extern char *program_invocation_name;
extern char *program_invocation_short_name;
extern char *__progname;

int main(int argc, char* argv[]) {
    // Get pid and name
    pid_t pid = getpid();
    printf("pid: %d\n", pid);
    printf("argv[0]: %s\n", argv[0]);
    printf("program_invocation_name: %s\n", program_invocation_name);
    printf("program_invocation_short_name: %s\n", program_invocation_short_name);
    printf("__progname: %s\n", __progname);
    printf("\n");
    
    // Perform some random operations
    printf("Perform random time comsuming operations...\n");
    srand(time(NULL));
    int r = 0;
    while (r < 10000000) {
        r = rand();
    }
    int sum = 0;
    for (int i = 0; i < r; i++) {
        sum += i;
    }
    printf("\n");

    // Get resource usage of current process
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    // Print usage info
    printf("ru_utime.tv_sec: %ld\n", usage.ru_utime.tv_sec);
    printf("ru_utime.tv_usec: %ld\n", usage.ru_utime.tv_usec);
    printf("ru_stime.tv_sec: %ld\n", usage.ru_stime.tv_sec);
    printf("ru_stime.tv_usec: %ld\n", usage.ru_stime.tv_usec);
    printf("ru_maxrss (kb): %ld\n", usage.ru_maxrss);
    // Unmaintained
    /* printf("ru_ixrss: %ld\n", usage.ru_ixrss); */
    /* printf("ru_idrss: %ld\n", usage.ru_idrss); */
    /* printf("ru_isrss: %ld\n", usage.ru_isrss); */
    printf("ru_minflt: %ld\n", usage.ru_minflt);
    printf("ru_majflt: %ld\n", usage.ru_majflt);
    // Unmaintained
    /* printf("ru_nswap: %ld\n", usage.ru_nswap); */
    printf("ru_inblock: %ld\n", usage.ru_inblock);
    printf("ru_oublock: %ld\n", usage.ru_oublock);
    // Unmaintained
    /* printf("ru_msgsnd: %ld\n", usage.ru_msgsnd); */
    /* printf("ru_msgrcv: %ld\n", usage.ru_msgrcv); */
    /* printf("ru_nsignals: %ld\n", usage.ru_nsignals); */
    printf("ru_nvcsw: %ld\n", usage.ru_nvcsw);
    printf("ru_nivcsw: %ld\n", usage.ru_nivcsw);

    return 0;
}
