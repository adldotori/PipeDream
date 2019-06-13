# AILAB

This repository is what I develop in AI lab.

<strong><u> GOAL : Parallel Training in PipeDream </u></strong>

(Please compile in Linux instead of MAC OS X)

## First step. Socket Programming
Two computers connect with socket.

#### HOW TO EXECUTE

go to tag : <strong><first_step></strong>

1. add code "#define NOT_AIO" in both files.
2. gcc -o server server.c -lrt
3. ./server
4. gcc -o client client.c -lrt
5. ./client output

## Second step. AIO Socket Programming
Two computers connect with socket and communicates in the way of Asynchronous non-blocking I/O instead Synchronous blocking I/O. 

