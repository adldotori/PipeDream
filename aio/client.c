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
#define BUFSIZE 1024
//#define NOT_AIO

const char CLIENT_IP[20] = "127.0.0.1";
const int CLIENT_PORT = 9995;

int main(int argc, char * argv[]){
    if(argc < 2) {
        printf("usage: %s <file's name>\n",argv[0]);
        exit(1);
    }
    struct sockaddr_in server, client;
    clock_t start,end;
    start = clock();
    struct aiocb my_aiocb;
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1)
    {
        printf("socket ERROR\n");
        exit(1);
    }
    bzero((char *)&server, sizeof(struct sockaddr_in));
    server.sin_family = AF_INET;
    server.sin_port = htons(CLIENT_PORT);
    server.sin_addr.s_addr = inet_addr(CLIENT_IP);
    if(connect(client_socket, (struct sockaddr *)&server, sizeof(server)) == -1)
    {
        printf("connect() ERROR\n");
        exit(1);
    }
    printf("client socket = [%d]\n",client_socket);
    
    end = clock();
    printf("connect socket : %.3fms\n",1000*(float)(end - start)/CLOCKS_PER_SEC);
    
    char buf[FILESIZE];
    
#ifdef NOT_AIO
    printf("read io\n");
    start = clock();
    int cnt=0,ret,rd_bytes=0;
    while((ret = read(client_socket,buf+(cnt++)*BUFSIZE,BUFSIZE)) > 0){
        rd_bytes+=ret;
//        printf("%d times read ... (%dbytes)\n",cnt,rd_bytes);
    }
    end = clock();
    printf("read file : %.3fms\n",1000*(float)(end - start)/CLOCKS_PER_SEC);

#else
    printf("read aio\n");
    
    int fd = open( "file.txt", O_RDWR );
    if (fd < 0) perror("open");
    
    int cnt=0;
    while(1){
        my_aiocb.aio_buf = buf+cnt*BUFSIZE;
        if (!my_aiocb.aio_buf) perror("malloc");

        my_aiocb.aio_fildes = client_socket;
        my_aiocb.aio_nbytes = BUFSIZE;
        my_aiocb.aio_offset = (cnt++)*BUFSIZE;
        
        int ret = aio_read(&my_aiocb);
        if (ret < 0) {
            perror("aio_read");
            printf("errno:%d\n",errno);
        }
        while ( aio_error( &my_aiocb ) == EINPROGRESS ) ;
        if ((ret = aio_return( &my_aiocb )) > 0)
        {
            printf("ret [%d]\n", ret);
        }
        else
        {
            printf("EOF\n");
            break;
        }
    }
#endif
    printf("%d %d\n",fd, client_socket);
    close(client_socket);
    printf("strlen(buf) : %lu\n",strlen(buf));
    int fd1 = open(argv[1] ,O_RDWR | O_CREAT, 0644);
    write(fd1,buf,strlen(buf));
    close(fd1);
}


