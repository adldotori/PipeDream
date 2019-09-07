// Pipeline Parallelism
#include <iostream>
#include <algorithm>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "aiocb.h"

#define MAX(a, b) (a) > (b) ? (a) : (b)
#define SQR(a) (a) * (a)
#define LEARNING_RATE 0.0001
#define DATA_SET 60000
#define TEST_DATA_SET 10000
#define BATCH_SIZE 1000
#define BUFSIZE 40000
#define MAXSIZE 5000000
#define OUT_SIZE 10 // the number of final class
#define AIOCB_NUM 20 // the number of communication module
#define EPOCH 1
using namespace std;

enum active_mode
{
    sigmoid = 0,
    ReLU,
    softmax
};
enum layer_type
{
    Input = 0,
    Hidden,
    Output
};

char ip[20] = "127.0.0.1";
int port = 1597;

class Layer
{
private:
    int in, out, len; // input node, output node, cnt of data
    int batch_size, tot_batch;
    double **w, *b;
    double *input, *output; // data*in, data*out
    double *predict;
    double *pre_pardiff;
    int before_socket, after_socket;
    enum active_mode active;
    enum layer_type layer_type;
    struct aiocb ** recv_aiocb;

    void activation(int k, double *val)
    {
        double *ret = new double[out];
        switch (active)
        {
        case sigmoid:
            for (int i = 0; i < out; i++)
                ret[i] = val[i] / (1 + exp(-val[i]));
            break;
        case ReLU:
            for (int i = 0; i < out; i++)
            {
                if (val[i] < 0)
                    ret[i] = 0;
                else
                    ret[i] = val[i];
            }
            break;
        case softmax:
            double sum = 0, avg = 0;
            double *max = new double[out];
            for (int i = 0; i < out; i++)
            {
                avg += val[i];
            }
            avg /= out;
            for (int i = 0; i < out; i++)
            {
                val[i] = val[i] - avg;
                if (val[i] < -300)
                    val[i] = -300;
                else if (val[i] > 300)
                    val[i] = 300;
                max[i] = exp(val[i]);
                sum += max[i];
            }
            for (int i = 0; i < out; i++)
            {
                ret[i] = max[i] / sum;
                if (ret[i] < exp(-30))
                    ret[i] = exp(-30);
                if (ret[i] == 1)
                    ret[i] = 1 - exp(-30);
            }
            break;
        }
        for (int i = 0; i < out; i++)
        {
            predict[k * out + i] = ret[i];
        }
    }

    void forwardProp(int batch)
    {
        // cout << "fw : " << batch << endl;
        for (int k = batch * batch_size; k < (batch + 1) * batch_size; k++)
        {
            double *ret = new double[out];
            for (int i = 0; i < out; i++)
            {
                for (int j = 0; j < in; j++)
                {
                    ret[i] += w[j][i] * input[k * in + j];
                }
                ret[i] += b[i];
            }
            activation(k, ret);
        }
    }

    double cost(void) // cross_entropy
    {
        double cost = 0;
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < out; j++)
            {
                cost -= output[i * out + j] * log(predict[i * out + j]) - (1 - output[i * out + j]) * log(1 - predict[i * out + j]);
            }
        }
        cost /= len;

        return cost;
    }

    void backwardProp(int batch)
    {
        // cout << "bw : " << batch << endl;
        for (int k = batch * batch_size; k < (batch + 1) * batch_size; k++)
        {
            for (int j = 0; j < out; j++)
            {
                int out_cnt = k * out + j;
                for (int i = 0; i < in; i++)
                {
                    int in_cnt = k * in + i;
                    w[i][j] -= LEARNING_RATE * pre_pardiff[out_cnt] * input[in_cnt];
                }
                b[j] -= LEARNING_RATE * pre_pardiff[out_cnt];
            }
        }
    }

    void prediction(void)
    {
        int correct = 0;
        for (int i = 0; i < len; i++)
        {
            int max = max_element(predict + i * out, predict + (i + 1) * out) - (predict + i * out);
            if (output[i * out + max] == 1)
                correct++;
        }
        cout << "Correct Rate : " << (double)correct / len << endl;
        cout << endl;
    }

    struct aiocb * recvBefore(int batch)
    {
        if (before_socket == -1)
            return NULL;
        int cnt = 0, ret, rd_bytes = 0, max_cnt = (batch_size * in * 8 - 1) / BUFSIZE + 1;
        struct aiocb *final_aiocb = (struct aiocb *)malloc(sizeof(struct aiocb));
        while (cnt < max_cnt)
        {
            struct aiocb *my_aiocb = new_aiocb(before_socket, (double *)((char *)(input + batch * batch_size * in) + cnt++ * BUFSIZE), BUFSIZE);
            aio_read(my_aiocb);
            final_aiocb = my_aiocb;
        }
        return final_aiocb;
    }

    void sendAfter(int batch)
    {
        if (after_socket == -1)
            return;
        struct aiocb *my_aiocb = new_aiocb(after_socket, predict + batch * batch_size * out, batch_size * out * 8);
        aio_write(my_aiocb);
    }

    struct aiocb * recvAfter(int batch)
    {
        if (after_socket == -1)
            return NULL;
        int cnt = 0, ret, rd_bytes = 0, max_cnt = (batch_size * out * 8 - 1) / BUFSIZE + 1;
        struct aiocb *final_aiocb = (struct aiocb *)malloc(sizeof(struct aiocb));
        while (cnt < max_cnt)
        {
            struct aiocb *my_aiocb = new_aiocb(after_socket, (double *)((char *)(pre_pardiff + batch * batch_size * out) + cnt++ * BUFSIZE), BUFSIZE);
            aio_read(my_aiocb);
            final_aiocb = my_aiocb;
        }
        return final_aiocb;
    }

    void sendBefore(int batch)
    {
        if (before_socket == -1)
            return;
        double *post_pardiff = new double[batch_size * in];
        for (int i = 0; i < batch_size * in; i++)
        {
            post_pardiff[i] = 0;
        }
        for (int k = batch * batch_size; k < (batch + 1) * batch_size; k++)
        {
            for (int i = 0; i < in; i++)
            {
                for (int j = 0; j < out; j++)
                {
                    int in_cnt = k * in + i;
                    int in_cnt_p = (k - batch * batch_size) * in + i;
                    int out_cnt = k * out + j;
                    switch (active)
                    {
                    case sigmoid:
                    case softmax:
                        post_pardiff[in_cnt_p] += pre_pardiff[out_cnt] * w[i][j] * input[in_cnt] * (1 - input[in_cnt]);
                        break;
                    case ReLU:
                        if (input[in_cnt] > 0)
                            post_pardiff[in_cnt_p] += pre_pardiff[out_cnt] * w[i][j];
                        break;
                    }
                }
            }
        }
        struct aiocb *my_aiocb = new_aiocb(before_socket, post_pardiff, batch_size * in * 8);
        aio_write(my_aiocb);
    }

    double gaussianRandom(void)
    {
        double v1, v2, s;
        do
        {
            v1 = 2 * ((double)rand() / RAND_MAX) - 1; // -1.0 ~ 1.0
            v2 = 2 * ((double)rand() / RAND_MAX) - 1; // -1.0 ~ 1.0
            s = v1 * v1 + v2 * v2;
        } while (s >= 1 || s == 0);

        s = sqrt((-2 * log(s)) / s);

        return v1 * s;
    }

    void openPort(void)
    {
        int server_socket = socket(PF_INET, SOCK_STREAM, 0);
        if (server_socket == -1)
        {
            printf("[-] socket ERROR\n");
            exit(1);
        }
        struct sockaddr_in *server = new_server(ip, port), client;
        if (bind(server_socket, (struct sockaddr *)server, sizeof(struct sockaddr_in)) == -1)
        {
            printf("[-] bind() ERROR\n");
            exit(1);
        }
        if (listen(server_socket, 5) == -1)
        {
            printf("[-] listen() ERROR\n");
            exit(1);
        }
        socklen_t client_size = sizeof(struct sockaddr_in);
        before_socket = accept(server_socket, (struct sockaddr *)&client, &client_size);
        if (before_socket == -1)
        {
            printf("[-] accept() ERROR\n");
            exit(1);
        }
        printf("[+] connect before_socket [%d]\n", before_socket);
    }

    void connNext(void)
    {
        char next_ip[20];
        int next_port;
        while (1)
        {
            cout << "Next Layer's ip : ";
            cin >> next_ip;
            cout << "Next Layer's port : ";
            cin >> next_port;
            after_socket = socket(AF_INET, SOCK_STREAM, 0);
            if (after_socket == -1)
            {
                printf("[-] socket ERROR\n");
                continue;
            }
            struct sockaddr_in *server = new_server(next_ip, next_port), client;
            if (connect(after_socket, (struct sockaddr *)server, sizeof(struct sockaddr_in)) == -1)
            {
                printf("[-] connect() ERROR\n");
                continue;
            }
            printf("[+] client after_socket [%d]\n", after_socket);
            break;
        }
        cout << "connect Completed!" << endl;
    }

    void forward(int batch)
    {
        forwardProp(batch);
        sendAfter(batch);
    }

    void backward(int batch)
    {
        if (layer_type == Output)
        {
            for (int i = batch * batch_size; i < (batch + 1) * batch_size; i++)
            {
                for (int j = 0; j < out; j++)
                {
                    int out_cnt = i * out + j;
                    pre_pardiff[out_cnt] = predict[out_cnt] - output[out_cnt];
                }
            }
        }
        backwardProp(batch);
        sendBefore(batch);
        if (layer_type == Output && batch == tot_batch - 1)
        {
            cout << "COST : " << cost() << endl;
            prediction();
        }
    }

public:
    Layer(int in_, int out_, enum active_mode active, enum layer_type layer_type)
    {
        // initialization
        in = in_;
        out = out_;
        output = NULL;
        this->active = active;
        this->layer_type = layer_type;
        before_socket = -1;
        after_socket = -1;
        srand(time(NULL));
        w = new double *[in];
        for (int i = 0; i < in; i++)
        {
            w[i] = new double[out];
            if (active == ReLU) // He initialization
            {
                for (int j = 0; j < out; j++)
                {
                    w[i][j] = gaussianRandom() / sqrt(in / 2);
                }
            }
            else // Xavier initialization
            {
                for (int j = 0; j < out; j++)
                {
                    w[i][j] = gaussianRandom() / sqrt(in);
                }
            }
        }
        b = new double[out];
        for (int i = 0; i < out; i++)
            b[i] = 0;
        if (layer_type != Output)
            connNext();
        if (layer_type != Input)
        {
            openPort();
        }
    }

    void getData(double *input, double *output, int len = DATA_SET)
    {
        this->len = len;
        batch_size = BATCH_SIZE;
        tot_batch = (len - 1) / batch_size + 1;
        this->input = new double[len * in];
        predict = new double[len * out];
        pre_pardiff = new double[len * out];
        recv_aiocb = new struct aiocb * [AIOCB_NUM];
        if (input != NULL && output != NULL)
        {
            this->input = input;
            this->output = output;
        }
        if (output == NULL)
        {
            this->output = new double[len * OUT_SIZE];
            int cnt = 0, ret, rd_bytes = 0;
            while (rd_bytes < len * OUT_SIZE * 8)
            {
                struct aiocb *my_aiocb = new_aiocb(before_socket, (double *)((char *)this->output + rd_bytes), min(len * OUT_SIZE * 8 - rd_bytes, BUFSIZE));
                aio_read(my_aiocb);
                while (my_aiocb->aio_fildes == before_socket)
                {
                }
                ret = aio_return(my_aiocb);
                rd_bytes += ret;
                cout << rd_bytes << ' ';
                if (ret < 0)
                {
                    cout << "ERROR!" << endl;
                    exit(1);
                }
                free(my_aiocb);
            }
        }
        if (layer_type != Output)
        {
            struct aiocb *my_aiocb = new_aiocb(after_socket, this->output, len * OUT_SIZE * 8);
            aio_write(my_aiocb);
            while (my_aiocb->aio_fildes == after_socket)
            {
            }
            free(my_aiocb);
        }
    }

    void training(int step)
    {
        for (int i = 0; i < step; i++)
        {
            cout << "training " << i + 1 << endl;
            int batch = 0, mach_idx;
            if (layer_type == Output)
                mach_idx = 0;
            else
                mach_idx = 1;
            for (int batch = 0; batch < tot_batch + 2 * mach_idx + 1; batch++)
            {
                int f_batch = batch - 1;
                int b_batch = batch - 2 * mach_idx - 1;

                // recvAfter(i - 2k - 1)
                if (b_batch >= 0 && b_batch < tot_batch)
                    recv_aiocb[(2 * b_batch + 1)%AIOCB_NUM] = recvAfter(b_batch);

                // forward(i - 1)
                if (f_batch >= 0 && f_batch < tot_batch)
                {
                    if (recv_aiocb[(2 * f_batch)%AIOCB_NUM]) {
                        while (recv_aiocb[(2 * f_batch)%AIOCB_NUM]->aio_fildes == before_socket)
                        {
                        }
                    }
                    forward(f_batch);
                }

                // recvBefore(i)
                if (batch >= 0 && batch < tot_batch)
                    recv_aiocb[(2 * batch)%AIOCB_NUM] = recvBefore(batch);

                // backward(i - 2k -1)
                if (b_batch >= 0 && b_batch < tot_batch)
                {
                    if (recv_aiocb[(2 * b_batch + 1)%AIOCB_NUM])
                        while (recv_aiocb[(2 * b_batch + 1)%AIOCB_NUM]->aio_fildes == after_socket)
                        {
                        }
                    backward(b_batch);
                }
            }
        }
    }

    void test(void)
    {
        cout << "TEST" << endl;
        batch_size = len / 10;
        tot_batch = 10;
        for (int i = 0; i < tot_batch; i++)
        {
            cout << i;
            struct aiocb * test_aiocb = recvBefore(i);
            if (test_aiocb)
                while (test_aiocb->aio_fildes == before_socket)
                {
                }
            forward(i);
            if (layer_type == Output && i == tot_batch - 1)
            {
                prediction();
            }
        }
    }

    void finish(void)
    {
        close(after_socket);
        close(before_socket);
    }
};

void download(double *input[], double *output[])
{
    unsigned int cnt;
    mnist_data *data;
    int ret = mnist_load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &data, &cnt);
    if (ret)
    {
        cout << "An error occured: " << ret << endl;
        exit(1);
    }
    int tmp = 0;
    for (int i = 0; i < DATA_SET; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            for (int k = 0; k < 28; k++)
            {
                *(*input + tmp++) = data[i].data[j][k];
            }
        }
        *(*output + i * 10 + data[i].label) = 1;
    }
    return;
}

void download_test(double *input[], double *output[])
{
    unsigned int cnt;
    mnist_data *data;
    int ret = mnist_load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &data, &cnt);
    if (ret)
    {
        cout << "An error occured: " << ret << endl;
    }
    int tmp = 0;
    for (int i = 0; i < TEST_DATA_SET; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            for (int k = 0; k < 28; k++)
            {
                *(*input + tmp++) = data[i].data[j][k];
            }
        }
        *(*output + i * 10 + data[i].label) = 1;
    }
    return;
}

int main(int argc, char **argv)
{
    int ch, count = EPOCH;
    while ((ch = getopt(argc, argv, "i:p:l:")) != -1)
    {
        switch (ch)
        {
        case 'i':
            strcpy(ip, optarg);
            break;
        case 'p':
            port = atoi(optarg);
            break;
        case 'l':
            if (!strcmp(optarg, "input"))
            {
                double *input = new double[784 * DATA_SET];
                double *output = new double[10 * DATA_SET];
                double *test_input = new double[784 * TEST_DATA_SET];
                double *test_output = new double[10 * TEST_DATA_SET];
                download(&input, &output);
                download_test(&test_input, &test_output);
                Layer layer(784, 256, ReLU, Input);

                layer.getData(input, output);
                layer.training(count);

                if (TEST_DATA_SET > 0)
                {
                    layer.getData(test_input, test_output, TEST_DATA_SET);
                    layer.test();
                }
                layer.finish();
            }
            else if (!strcmp(optarg, "hidden"))
            {
                Layer layer(256, 256, ReLU, Hidden);
                layer.getData(NULL, NULL);
                layer.training(count);

                if (TEST_DATA_SET > 0)
                {
                    layer.getData(NULL, NULL, TEST_DATA_SET);
                    layer.test();
                }
                layer.finish();
            }
            else if (!strcmp(optarg, "output"))
            {
                Layer layer(256, 10, softmax, Output);
                layer.getData(NULL, NULL);
                layer.training(count);

                if (TEST_DATA_SET > 0)
                {
                    layer.getData(NULL, NULL, TEST_DATA_SET);
                    layer.test();
                }
                layer.finish();
            }
            else
            {
                cout << "Only hidden, output can be -l option's argument." << endl;
            }
        }
    }
}
