#include "aiocb.h"

int main(int argc, char * argv[]){
    char ip[20]="127.0.0.1";
    int port = 9999;
    if(argc != 3){
         printf("default ip : 127.0.0.1 , port : 9999\n");
    }
    else {
        strcpy(ip,argv[1]);
        port = atoi(argv[2]);
    }
 
    int fd = open("file.txt",O_RDONLY);
    char buf[FILESIZE];
    read(fd,buf,FILESIZE);
    close(fd);

    int server_socket = socket(PF_INET, SOCK_STREAM, 0);
    if (server_socket == -1)
    {
        printf("[-] socket ERROR\n");
        exit(1);
    }
    struct sockaddr_in * server = new_server(ip,port), client;
    if(bind(server_socket, (struct sockaddr *)server, sizeof(struct sockaddr_in)) == -1)
    {
        printf("[-] bind() ERROR\n");
        exit(1);
    }
    if(listen(server_socket, 5) == -1)
    {
        printf( "[-] listen() ERROR\n");
        exit(1);
    }

    while(1){
        clock_t start,end;
        start = clock();
        socklen_t client_size = sizeof(struct sockaddr_in);
        int client_socket = accept(server_socket, (struct sockaddr *)&client, &client_size);
        if(client_socket == -1)
        {
            printf( "[-] accept() ERROR\n");
            exit(1);
        }
        end = clock();
        printf("[+] connect socket [%d] : %.3fms\n",client_socket,1000*(float)(end - start)/CLOCKS_PER_SEC);
        
        start = clock();
#ifdef NOT_AIO
        printf("[+] write io\n");
        write(client_socket, buf, strlen(buf));
#else
        printf("[+] write aio\n");
        int cnt = 0;
        struct aiocb * my_aiocb = new_aiocb(client_socket, buf, 0, strlen(buf));
        int ret = aio_write(my_aiocb);
        if(ret < 0) perror("aio_write");
        while(my_aiocb->aio_fildes == client_socket){
            printf("waiting...\n");
            sleep(1);
        }
#endif
        end = clock();
        printf("[+] write file : %.3fms\n\n",1000*(float)(end - start)/CLOCKS_PER_SEC);
        close(client_socket);
    }
    close(server_socket);
}
