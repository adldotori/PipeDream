#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <aio.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <math.h>
#define BUFSIZE 1000

int main(){
    // clock_t start,end;
    
    // start = clock();
    // for(int i=0;i<10000;i++){
    //     printf("%f",pow(0.9999,100000000));
    // }
    // end = clock();
    // printf("\ntime: %f",1000*(float)(end - start)/CLOCKS_PER_SEC);
    for(int i=0;i<1000;i++)
        printf("%f\n",sqrt(1-pow(0.999, i)) / (1-pow(0.9, i)));
}
