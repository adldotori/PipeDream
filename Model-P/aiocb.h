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
#define FILESIZE 5000000
//#define NOT_AIO

struct aiocb * new_aiocb(int fd, char *buf,int cnt, int buf_size);
void aio_handler(sigval_t sigval);
struct sockaddr_in * new_server(char *ip, int port);
