# AILAB

This repository is what I developed in AI lab.

**GOAL : Parallel Training in PipeDream**

(Please compile in Linux instead of MAC OS X)

## First step. Socket Programming
Two computers connect with socket.

#### *HOW TO EXECUTE*
go to tag : **<first_step>**
1. go to aio folder
2. add code "#define NOT_AIO" in both files.
3. gcc -o server server.c -lrt
4. ./server
5. gcc -o client client.c -lrt
6. ./client output

## Second step. AIO Socket Programming
Two computers connect with socket and communicates in the way of <strong>Asynchronous non-blocking I/O</strong> instead of <strong>Synchronous blocking I/O</strong>. 

#### *HOW TO EXECUTE*
go to tag : **<second_step>**
1. go to aio folder
2. gcc -o server server.c aiocb.c -lrt
3. ./server \<ip> \<port>
4. gcc -o client client.c aiocb.c -lrt
5. ./client <file\'s name> \<ip> \<port>

