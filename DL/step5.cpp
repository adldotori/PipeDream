// mnist(multi layer neural network)
#include <iostream>
#include <algorithm>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

#define MAX(a, b) (a) > (b) ? (a) : (b)
#define SQR(a) (a) * (a)
#define LEARNING_RATE 0.1
#define DATA_SET 1
#define how_cost 2     // 1:MSE, 2:ACE
#define how_optimize 2 // 1:gradient descent
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

class Network
{
private:
    int in, out, len; // input node, output node, cnt of data
    int batch_size;
    double **w, *b;
    double *input, *output; // data*in, data*out
    double *predict;
    double *next_weight;
    enum active_mode active;

    void activation(int data, double *val)
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
            double sum = 0;
            for (int i = 0; i < out; i++)
            {
                // cout << val[i] << " ";
                sum += exp(val[i]);
            }
            for (int i = 0; i < out; i++)
            {
                ret[i] = exp(val[i]) / sum;
                if (ret[i] < exp(-30))
                    ret[i] = exp(-30);
            }
            break;
        }
        for (int i = 0; i < out; i++)
            predict[data * out + i] = ret[i];
    }

    void forwardProp(void)
    {
        for (int data = 0; data < len; data++)
        {
            double *ret = new double[out];
            for (int i = 0; i < out; i++)
            {
                for (int j = 0; j < in; j++)
                {
                    ret[i] += w[j][i] * input[data * in + j];
                    if(i==0)
                        cout << w[j][i] << " " << input[data*in+j]<<endl;
                }
                ret[i] += b[i];
            }
            if (out == 10)
            {
                for (int i = 0; i < out; i++)
                    cout << ret[i] << " ";
                cout << endl;
            }
            activation(data, ret);
        }
    }

    double cost(void)
    {
        double cost = 0;
#if how_cost == 1 // MSE(mean square error)
        for (int i = batch * batch_size; i < (batch + 1) * batch_size; i++)
        {
            for (int j = 0; j < out; j++)
            {
                double val = predict[i][j] - output[i * out + j];
                cost += 0.5 * SQR(val);
            }
        }
        cost /= len;
#elif how_cost == 2 // ACE(average cross entropy)
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < out; j++)
            {
                cost -= output[i * out + j] * log(predict[i * out + j]);
                // cout << " cost :"  << predict[i*out+j];
            }
        }
        cost /= len;
#endif
        // etc ...
        return cost;
    }

    void backwardProp(void)
    {
        // cross-entropy
        switch (active)
        {
        case sigmoid:
            break;
        case ReLU:
            for (int bat = 0; bat < len / batch_size; bat++)
            {
                for (int j = 0; j < out; j++)
                {
                    for (int k = 0; k < batch_size; k++)
                    {
                        for (int i = 0; i < in; i++)
                        {
                            w[i][j] -= LEARNING_RATE * output[(bat * batch_size + k) * in + i] * next_weight[j];
                            // cout <<  output[(bat*batch_size+k)*in+i] << " ";
                        }
                        b[j] -= LEARNING_RATE * next_weight[j];
                        // cout << b[j];
                    }
                    // cout << j << next_weight[j] << ' ';
                }
            }
            break;
            // for(int k = 0; k < batch_size; k++)
            //     {
            //         int positive = predict[(bat*batch_size+k)*out+j] > 0;
            //         for (int i = 0; i < in; i++)
            //         {
            //             w[i][j] -= LEARNING_RATE * positive * (-output[(bat*batch_size+k)*out+j]/predict[(bat*batch_size+k)*out+j]) * input[(bat*batch_size+k)*in+i];
            //         }
            //         b[j] -= LEARNING_RATE * positive * (-output[(bat*batch_size+k)*out+j]/predict[(bat*batch_size+k)*out+j];
            //     }

        case softmax:
            for (int bat = 0; bat < len / batch_size; bat++)
            {
                for (int j = 0; j < out; j++)
                {
                    for (int k = 0; k < batch_size; k++)
                    {
                        for (int i = 0; i < in; i++)
                        {
                            w[i][j] -= LEARNING_RATE * (predict[(bat * batch_size + k) * out + j] - output[(bat * batch_size + k) * out + j]) * input[(bat * batch_size + k) * in + i];
                            // cout << w[i][j] << " ";
                        }
                        b[j] -= LEARNING_RATE * (predict[(bat * batch_size + k) * out + j] - output[(bat * batch_size + k) * out + j]);
                        // cout << b[j];
                    }
                }
            }
        }
    }

    void prediction(void)
    {
        cout << "Prediction: " << endl;
        int correct = 0;
        for (int i = 0; i < len; i++)
        {
            int max = max_element(predict + i * out, predict + (i + 1) * out) - (predict + i * out);
            if (output[i * out + max] == 1)
                correct++;
        }
        cout << "Correct Rate : " << (double)correct / len << endl
             << endl;
    }

    void send_before(void)
    {
        if (before == NULL)
            return;
        this->before->next_weight = new double[in];
        for (int i = 0; i < in; i++)
        {
            double val = 0;
            for (int j = 0; j < out; j++)
            {
                for (int k = 0; k < len; k++)
                {
                    if (input[k * in + i] > 0 && predict[k * out + j] > 0)
                        val -= output[k * out + j] * w[i][j] / predict[k * out + j];
                    // cout << predict[k * out + j] << endl;
                }
            }
            this->before->next_weight[i] = val;
            // cout << val << " ";
        }
    }

    void send_after(void)
    {
        if (after == NULL)
            return;
        this->after->getData(predict, output);
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

public:
    Network *before, *after;
    Network(int in_, int out_, int len, enum active_mode active)
    {
        // initialization
        srand(time(NULL));
        in = in_;
        out = out_;
        this->len = len;
        this->active = active;
        w = new double *[in];

        srand(time(NULL));
        for (int i = 0; i < in; i++)
        {
            w[i] = new double[out];
            if (active == ReLU) // He initialization
            {
                for (int j = 0; j < out; j++)
                {
                    w[i][j] = gaussianRandom() / sqrt(in/2);
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
        before = NULL;
        after = NULL;
        batch_size = 10;
        if (len < batch_size)
            batch_size = 1;
    }

    void connect(Network *other)
    {
        this->after = other;
        other->before = this;
    }

    void getData(double *input, double *output)
    {
        this->input = input;
        this->output = output;
    }

    void training(int step)
    {
        for (int i = 0; i < step; i++)
        {
            cout << "training " << i + 1 << endl;
            forwardProp();
            send_after();
            cout << "inout" << in << out;
            if (after != NULL)
                after->training(1);
            if (out == 10)
            {
                for (int j = 0; j < in; j++)
                {
                    cout << input[j] << " ";
                }
                for (int j = 0; j < out; j++)
                {
                    cout << predict[j] << " ";
                }
                for (int j = 0; j < out; j++)
                {
                    cout << output[j];
                }
            }
            backwardProp();
            if(active == softmax)
                cout << "COST : " << cost() << endl;
            send_before();
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
    }
    else
    {
        cout << "image count: " << cnt << endl;
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

int main()
{
    double *input = new double[784 * DATA_SET];
    double *output = new double[784 * DATA_SET];
    download(&input, &output);

    Network net1(784, 100, DATA_SET, ReLU);
    Network net2(100, 10, DATA_SET, softmax);
    net1.connect(&net2);

    net1.getData(input, output);
    net1.training(10);
}