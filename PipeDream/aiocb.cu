#include "aiocb.h"

struct aiocb * new_aiocb(int fd, double *buf, int buf_size){
    struct aiocb *my_aiocb = (struct aiocb *)malloc(sizeof(struct aiocb));
    bzero( (double *)my_aiocb, sizeof(struct aiocb) );

    my_aiocb->aio_fildes     = fd;
    my_aiocb->aio_lio_opcode = LIO_READ; /* for lio_listio */
    my_aiocb->aio_buf        = buf;
    my_aiocb->aio_nbytes     = buf_size;
    my_aiocb->aio_offset     = 0;

    my_aiocb->aio_sigevent.sigev_notify            = SIGEV_THREAD;
    my_aiocb->aio_sigevent.sigev_signo             = 0;
    my_aiocb->aio_sigevent.sigev_notify_function   = aio_handler;
    my_aiocb->aio_sigevent.sigev_notify_attributes = NULL;
    my_aiocb->aio_sigevent.sigev_value.sival_ptr   = my_aiocb;

    return my_aiocb;
}

#if LINUX==CentOS
void aio_handler(sigval_t sigval)
#elif LINUX==ubuntu
void aio_handler(__sigval_t sigval)
#elif LINUX==MacOS
void aio_handler(sigval sigval)
#endif
{
    struct aiocb * my_aiocb = (struct aiocb *)sigval.sival_ptr;
    int fd = my_aiocb->aio_fildes;
    if (aio_error( my_aiocb ) == 0) {
        // printf("ret:%d\n",aio_return(my_aiocb));
        my_aiocb->aio_fildes = 0;
    }
    else {
	    my_aiocb->aio_fildes = -1;
    }
    return;
}

struct sockaddr_in * new_server(char *ip, int port){
    struct sockaddr_in * server = (struct sockaddr_in *)malloc(sizeof(struct sockaddr_in));
    bzero((char *)server, sizeof(struct sockaddr_in));
    server->sin_family = AF_INET;
    server->sin_port = htons(port);
    server->sin_addr.s_addr = inet_addr(ip);
    
    return server;
}
