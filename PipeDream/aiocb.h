#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#define CentOS 1
#define ubuntu 2
#define MacOS 3
#define LINUX ubuntu
//#define NOT_AIO

struct aiocb * new_aiocb(int fd, double *buf, int buf_size);
#if LINUX==CentOS
void aio_handler(sigval_t sigval);
#elif LINUX==ubuntu
void aio_handler(__sigval_t sigval);
#elif LINUX==MacOS
void aio_handler(sigval sigval);
#endif
struct sockaddr_in * new_server(char *ip, int port);
