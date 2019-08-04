// multi computer
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
#define LEARNING_RATE 0.001
#define DATA_SET 60000
#define BATCH_SIZE 100
#define BUFSIZE 20000
#define OUT_SIZE 10
using namespace std;

enum mode
{
    train = 0,
    test
};
enum active_mode
{
    sigmoid = 0,
    ReLU,
    softmax
};
enum layer_type
{
    Hidden = 0,
    Output
};

char ip[20];
int port;

class Layer
{
private:
    int in, out, len; // input node, output node, cnt of data
    int batch_size;
    double **w, *b;
    double *input, *output; // data*in, data*out
    double *predict;
    double *pre_pardiff;
    int before_socket, after_socket;
    enum active_mode active;
    enum layer_type layer_type;

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

    void recvBefore(int batch)
    {
        if (before_socket == -1)
            return;
        int cnt = 0, ret, rd_bytes = 0;
        while (rd_bytes < batch_size * in * 8)
        {
            ret = read(before_socket, input + batch * batch_size * in + (cnt++) * BUFSIZE, BUFSIZE);
            rd_bytes += ret;
            if (ret == 0)
                break;
        }
    }

    void sendAfter(int batch)
    {
        if (after_socket == -1)
            return;
        write(after_socket, predict + batch * batch_size * out, batch_size * out * 8);
    }

    void recvAfter(int batch)
    {
        if (after_socket == -1)
            return;
        int cnt = 0, ret, rd_bytes = 0;
        while (rd_bytes < batch_size * out * 8)
        {
            ret = read(after_socket, pre_pardiff + batch * batch_size * out + (cnt++) * BUFSIZE, BUFSIZE);
            rd_bytes += ret;
            if (ret == 0)
                break;
        }
    }

    void sendBefore(int batch)
    {
        if (before_socket == -1)
            return;

        double *post_pardiff = new double[batch_size * in];

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
        write(before_socket, post_pardiff, batch_size * in * 8);
    }

    double gaussianRandom(void)
    {
        double v1, v2, s;
        do
        {
            v1 = 2 * ((double)rand() / RAND_MAX) - 1; // -1.0 ~ 1.0 까지의 값
            v2 = 2 * ((double)rand() / RAND_MAX) - 1; // -1.0 ~ 1.0 까지의 값
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

public:
    Layer(int in_, int out_, int len, enum active_mode active, enum layer_type layer_type)
    {
        // initialization
        srand(time(NULL));
        in = in_;
        out = out_;
        this->len = len;
        input = new double[len * in];
        output = NULL;
        this->active = active;
        this->layer_type = layer_type;
        before_socket = -1;
        after_socket = -1;
        w = new double *[in];
        srand(time(NULL));
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
        predict = new double[len * out];
        pre_pardiff = new double[len * out];
        batch_size = BATCH_SIZE;
        if (len < batch_size)
            batch_size = 1;
        if (layer_type != Output)
            connNext();
        if (in != 784)
        {
            openPort();
            sendOutput();
        }
    }

    ~Layer()
    {
        close(after_socket);
        close(before_socket);
    }

    void getData(double *input, double *output)
    {
        this->input = input;
        this->output = output;
        sendOutput();
    }

    void sendOutput(void)
    {
        if (output == NULL)
        {
            output = new double[len * OUT_SIZE];
            printf("[+] read io\n");
            int cnt = 0, ret, rd_bytes = 0;
            while (rd_bytes < len * OUT_SIZE * 8)
            {
                ret = read(before_socket, output + rd_bytes / 8, BUFSIZE);
                rd_bytes += ret;
                cout << cnt++ << " times read ... (" << rd_bytes << "bytes)" << endl;
            }
        }
        if (layer_type == Hidden)
        {
            printf("[+] write io\n");
            cout << write(after_socket, output, len * OUT_SIZE * 8) << endl;
        }
    }

    void batch_training(int batch)
    {
        cout << batch << endl;
        recvBefore(batch);
        forwardProp(batch);
        sendAfter(batch);
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
        recvAfter(batch);
        backwardProp(batch);
        sendBefore(batch);
        if (layer_type == Output && batch == len / batch_size - 1)
        {
            cout << "COST : " << cost() << endl;
            prediction();
        }
    }

    void training(int step)
    {
        for (int i = 0; i < step; i++)
        {
            cout << "training " << i + 1 << endl;
            for (int j = 0; j < len / batch_size; j++)
            {
                batch_training(j);
            }
        }
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

int main(int argc, char **argv)
{
    int ch, count = 3;
    while ((ch = getopt(argc, argv, "i:p:l:")) != -1)
    {
        switch (ch)
        {
        case 'i':
            strncpy(ip, optarg, strlen(optarg));
            break;
        case 'p':
            port = atoi(optarg);
            break;
        case 'l':
            if (!strcmp(optarg, "input"))
            {
                double *input = new double[784 * DATA_SET];
                double *output = new double[10 * DATA_SET];
                download(&input, &output);
                Layer layer(784, 256, DATA_SET, ReLU, Hidden);

                layer.getData(input, output);
                layer.training(count);
            }
            else if (!strcmp(optarg, "hidden"))
            {
                Layer layer(256, 256, DATA_SET, ReLU, Hidden);
                layer.training(count);
            }
            else if (!strcmp(optarg, "output"))
            {
                Layer layer(256, 10, DATA_SET, softmax, Output);
                layer.training(count);
            }
            else
            {
                cout << "Only hidden, output can be -l option's argument." << endl;
            }
        }
    }
}
