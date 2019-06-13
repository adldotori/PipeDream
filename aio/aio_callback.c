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
#include <signal.h>
#define FILESIZE 5000000
#define BUFSIZE 10000

void aio_handler(union sigval sigval);

struct aiocb * new_aiocb(int fd, char *buf, int cnt){
    struct aiocb *my_aiocb = malloc(sizeof(struct aiocb));
    bzero( (char *)my_aiocb, sizeof(struct aiocb) );
    
    my_aiocb->aio_fildes     = fd;
    my_aiocb->aio_lio_opcode = LIO_READ; /* for lio_listio */
    my_aiocb->aio_buf        = buf+cnt*BUFSIZE;
    my_aiocb->aio_nbytes     = BUFSIZE;
    my_aiocb->aio_offset     = cnt*BUFSIZE;
    
    my_aiocb->aio_sigevent.sigev_notify            = SIGEV_THREAD;
    my_aiocb->aio_sigevent.sigev_signo             = 0;
    my_aiocb->aio_sigevent.sigev_notify_function   = aio_handler;
    my_aiocb->aio_sigevent.sigev_notify_attributes = NULL;
    my_aiocb->aio_sigevent.sigev_value.sival_ptr   = my_aiocb;
    
    int ret = aio_read( my_aiocb );
    return my_aiocb;
}
void aio_handler(union sigval sigval){
    struct aiocb * my_aiocb = sigval.sival_ptr;
    printf("aio_handler\n");
    int fd = my_aiocb->aio_fildes,ret;
    if (aio_error( my_aiocb ) == 0) {
        int ret = aio_return( my_aiocb );
        printf("ret [%d]\n", ret);
	if(ret == 0) exit(1);
    }
    return;
}
int main(){
    int fd, ret;
    
    fd = open( "file.txt", O_RDONLY );
    if (fd < 0) perror("open");
    
    int cnt=0;
    char buf[FILESIZE];
    while(1){
        struct aiocb * my_aiocb = new_aiocb(fd, buf, cnt++);
    }
    while(1){
        printf("Still Waiting...\n");
        sleep(1);
    }
}


