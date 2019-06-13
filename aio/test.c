#include <stdio.h>
#include <stdlib.h>
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
#define BUFSIZE 1000

int main(){
    int fd, ret;
    struct aiocb my_aiocb;

    fd = open( "file.txt", O_RDONLY );
    if (fd < 0) perror("open");
    
    int cnt=0;
    char buf[40000];
    /* Zero out the aiocb structure (recommended) */
    bzero( (char *)&my_aiocb, sizeof(struct aiocb) );
    
    while(cnt<10){
        /* Allocate a data buffer for the aiocb request */
        my_aiocb.aio_buf = buf+cnt*BUFSIZE;
        if (!my_aiocb.aio_buf) perror("malloc");

        /* Initialize the necessary fields in the aiocb */
        my_aiocb.aio_fildes = fd;
        my_aiocb.aio_nbytes = BUFSIZE;
        my_aiocb.aio_offset = (cnt++)*BUFSIZE;

        ret = aio_read( &my_aiocb );
        if (ret < 0) perror("aio_read");
        while ( aio_error( &my_aiocb ) == EINPROGRESS ) ;

        if ((ret = aio_return( &my_aiocb )) > 0) {
            /* got ret bytes on the read */
            printf("%lu\n",strlen(buf));
        } else {
            printf("read failed\n");
            /* read failed, consult errno */
        }
    }
    printf("%s",buf);
}
