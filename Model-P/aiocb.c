#include "aiocb.h"

struct aiocb * new_aiocb(int fd, char *buf, int cnt, int buf_size){
    struct aiocb *my_aiocb = malloc(sizeof(struct aiocb));
    bzero( (char *)my_aiocb, sizeof(struct aiocb) );

    my_aiocb->aio_fildes     = fd;
    my_aiocb->aio_lio_opcode = LIO_READ; /* for lio_listio */
    my_aiocb->aio_buf        = buf+cnt*buf_size;
    my_aiocb->aio_nbytes     = buf_size;
    my_aiocb->aio_offset     = cnt*buf_size;

    my_aiocb->aio_sigevent.sigev_notify            = SIGEV_THREAD;
    my_aiocb->aio_sigevent.sigev_signo             = 0;
    my_aiocb->aio_sigevent.sigev_notify_function   = aio_handler;
    my_aiocb->aio_sigevent.sigev_notify_attributes = NULL;
    my_aiocb->aio_sigevent.sigev_value.sival_ptr   = my_aiocb;

    return my_aiocb;
}

void aio_handler(sigval_t sigval){
    struct aiocb * my_aiocb = sigval.sival_ptr;
    int fd = my_aiocb->aio_fildes,ret;
    if (aio_error( my_aiocb ) == 0) {
        int ret = aio_return( my_aiocb );
        printf("[+] %3dth aiocb's response [size: %5dbytes]\n",(int)(my_aiocb->aio_offset/my_aiocb->aio_nbytes)+1, ret);
        my_aiocb->aio_fildes = -1;
    }
    else perror("aio_handler");
    return;
}

struct sockaddr_in * new_server(char *ip, int port){
    struct sockaddr_in * server = malloc(sizeof(struct sockaddr_in));
    bzero((char *)server, sizeof(struct sockaddr_in));
    server->sin_family = AF_INET;
    server->sin_port = htons(port);
    server->sin_addr.s_addr = inet_addr(ip);
    
    return server;
}
