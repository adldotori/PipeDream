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

const char SERVER_IP[20] = "127.0.0.1";
const int SERVER_PORT = 9995;

int main(){
    int fd = open("file.txt",O_RDONLY);
    char buf[FILESIZE];
    read(fd,buf,FILESIZE);
    close(fd);
    
    struct sockaddr_in server, client;
    int server_socket = socket(PF_INET, SOCK_STREAM, 0);
    if (server_socket == -1)
    {
        printf("socket ERROR\n");
        exit(1);
    }
    bzero((char *)&server, sizeof(struct sockaddr_in));
    server.sin_family = AF_INET;
    server.sin_port = htons(SERVER_PORT);
    server.sin_addr.s_addr = inet_addr(SERVER_IP);
    if(bind(server_socket, (struct sockaddr *)&server, sizeof(server)) == -1)
    {
        printf("bind() ERROR\n");
        exit(1);
    }
    if(listen(server_socket, 5) == -1)
    {
        printf( "listen() ERROR\n");
        exit(1);
    }
    
    while(1){
        clock_t start,end;
        start = clock();
        socklen_t client_size = sizeof(struct sockaddr_in);
        int client_socket = accept(server_socket, (struct sockaddr *)&client, &client_size);
        if(client_socket == -1)
        {
            printf( "accept() ERROR\n");
            exit(1);
        }
        end = clock();
        printf("connect socket [%d] : %.3fms\n",client_socket,1000*(float)(end - start)/CLOCKS_PER_SEC);

#ifdef NOT_AIO 
        printf("write io\n");
        start = clock();
        write(client_socket, buf, strlen(buf));
        end = clock();
#else
        struct aiocb my_aiocb;
        printf("write aio\n");
        bzero( (char *)&my_aiocb, sizeof(struct aiocb) );
        my_aiocb.aio_buf = buf;
        my_aiocb.aio_fildes = client_socket;
        my_aiocb.aio_nbytes = strlen(buf);
        my_aiocb.aio_offset = 0;
        int ret = aio_write(&my_aiocb);
        if (ret < 0) {
            perror("aio_write");
            printf("errno:%d\n",errno);
        }
        while ( aio_error( &my_aiocb ) == EINPROGRESS );
        printf("%d",aio_error(&my_aiocb));
        if ((ret = aio_return( &my_aiocb )) > 0)
        {
            printf("ret [%d]\n", ret);
        }
        else
        {
            printf("write failed\n");
        }
        end = clock();
#endif
        printf("write file : %.3fms\n\n",1000*(float)(end - start)/CLOCKS_PER_SEC);
        close(client_socket);
    }
    close(server_socket);
}

