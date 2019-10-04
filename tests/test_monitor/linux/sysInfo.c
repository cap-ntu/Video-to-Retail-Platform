// Desc: Linux c program to get system info.
// Author: Zhou Shengsheng
// Date: 22/04/19
//
// References:
//

#include <stdio.h>
#include <sys/sysinfo.h>
 
int main()
{
    struct sysinfo si;
    sysinfo(&si);
    printf("Totalram: %ld\n", si.totalram);
    printf("Available: %ld\n", si.freeram);
    return 0;
}
