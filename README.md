# AILAB

This repository is what I develop in AI lab.

<strong> GOAL : Parallel Training in PipeDream </strong>

## First step. Socket Programming
Two computers connect with socket. 
tag:first_step
add code "define NOT_AIO" in both files.
1. gcc -o server server.c
2. ./server
3. gcc -o client client.c
4. ./client output

## Second step. AIO Socket Programming
Two computers connect with socket and communicates in the way of Asynchronous non-blocking I/O instead Synchronous blocking I/O. 
