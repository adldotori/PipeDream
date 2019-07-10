# AILAB

This repository is what I developed in AI lab.

**GOAL : Parallel Training in PipeDream**

(Please compile in Linux instead of MAC OS X)


## Level 1 : AIO

### 1st step. Socket Programming
Two computers connect with socket.

* HOW TO EXECUTE

    go to tag : **<aio_1st_step>**
    1. go to aio folder
    2. add code "#define NOT_AIO" in both files.
    3. gcc -o server server.c -lrt
    4. ./server
    5. gcc -o client client.c -lrt
    6. ./client output

### 2nd step. AIO Socket Programming

Two computers connect with socket and communicates in the way of <strong>Asynchronous non-blocking I/O</strong> instead of <strong>Synchronous blocking I/O</strong>. 

* HOW TO EXECUTE

    go to tag : **<aio_2nd_step>**
    1. go to aio folder
    2. gcc -o server server.c aiocb.c -lrt
    3. ./server \<ip> \<port>
    4. gcc -o client client.c aiocb.c -lrt
    5. ./client <file\'s name> \<ip> \<port>


## Level 2 : DL

### 1st step. Linear Regression
Simple Linear Regression with neuron class

* How to execute

    go to tag : **<DL_1st_step>**
    1. go to DL folder
    2. g++ -o step1 step1.cpp
    3. ./step1

### 2nd step. Multi Variable Linear Regression

Multi Variable Linear Regression with neuron class

* How to execute

    same before step

### 3rd step. Basic Neural Network

basic neural network(use softmax cross-entropy)

* How to execute

    same before step

### 4th step. MNIST(Basic Neural Network)

adapt basic neural network to mnist

The correct solution was fitted with a probability of <strong>93%</strong>.

* How to execute

    same before step

    you can change LEARNING_RATE, DATA_SET, batch_size


## Level 3 : GPU
---