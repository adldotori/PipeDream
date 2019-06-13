#include "aiocb.h"
#define BUFSIZE 80000

int main(int argc, char * argv[]){
    char ip[20]="127.0.0.1";
    int port = 9999;
    if(argc != 4){
         printf("usage: %s <file's name> <ip> <port>\n", argv[0]);
         printf("default ip : 127.0.0.1 , port : 9999\n");
    }
    else {
        if(argc < 2) exit(1);
        strcpy(ip,argv[2]);
        port = atoi(argv[3]);
    }
    clock_t start,end;
    start = clock();
    struct aiocb my_aiocb;
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1)
    {
        printf("[-] socket ERROR\n");
        exit(1);
    }
    struct sockaddr_in * server = new_server(ip,port), client;
    if(connect(client_socket, (struct sockaddr *)server, sizeof(struct sockaddr_in)) == -1)
    {
        printf("[-] connect() ERROR\n");
        exit(1);
    }
    printf("client socket = [%d]\n",client_socket);
    
    end = clock();
    printf("[+] connect socket : %.3fms\n",1000*(float)(end - start)/CLOCKS_PER_SEC);
    
    char buf[FILESIZE];
    start = clock();
#ifdef NOT_AIO
    printf("[+] read io\n");
    int cnt=0,ret,rd_bytes=0;
    while((ret = read(client_socket,buf+(cnt++)*BUFSIZE,BUFSIZE)) > 0){
        rd_bytes+=ret;
//        printf("%d times read ... (%dbytes)\n",cnt,rd_bytes);
    }
#else
    printf("[+] read aio\n");
    int cnt=0;
    struct aiocb * ck = malloc(sizeof(struct aiocb));
    ck->aio_fildes = client_socket;
   
    while(cnt<FILESIZE/BUFSIZE){
        struct aiocb * my_aiocb = new_aiocb(client_socket,buf,cnt++, BUFSIZE);
        int ret = aio_read(my_aiocb);
        if(ret < 0) perror("aio_read");
        ck = my_aiocb;
    }
    while(ck->aio_fildes == client_socket){
        printf("waiting...\n");
        sleep(1);
    }
#endif
    end = clock();
    printf("[+] read file : %.3fms\n",1000*(float)(end - start)/CLOCKS_PER_SEC);
    close(client_socket);
    printf("strlen(buf) : %lu\n",strlen(buf));
    int fd1 = open(argv[1] ,O_RDWR | O_CREAT, 0644);
    write(fd1,buf,strlen(buf));
    close(fd1);
}


