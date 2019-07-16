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
#define LEARNING_RATE 0.001
#define DATA_SET 1000
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

class Layer
{
private:
    int in, out, len; // input node, output node, cnt of data
    int batch_size;
    double **w, *b;
    double *input, *output; // data*in, data*out
    double *predict;
    int pre_out;
    double *pre_pardiff;
    double *post_pardiff;
    enum active_mode active;
    enum layer_type layer_type;

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
                if (ret[i] == 1)
                    ret[i] = 1 - exp(-30);
            }
            break;
        }
        for (int i = 0; i < out; i++) {
            predict[data * out + i] = ret[i];
            // cout << predict[data* out + i] << ' ';
        }
        // for(int i=0;i<out;i++)
        //     cout << output[data*out + i];
        // cout << endl;
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
                }
                ret[i] += b[i];
            }
            activation(data, ret);
        }
    }

    double cost(void) // cross_entropy
    {
        double cost = 0;
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < out; j++)
            {
                if(predict[i*out+j]<exp(-30) || predict[i*out+j]>=1) cout << predict[i*out+j]<< " ";
                cost -= output[i * out + j] * log(predict[i * out + j]) - (1- output[i * out + j]) * log(1 - predict[i * out + j]);
            }
        }
        cost /= len;

        return cost;
    }

    void backwardProp(void)
    {
        for (int bat = 0; bat < len / batch_size; bat++)
        {
            for (int k = 0; k < batch_size; k++)
            {
                for (int j = 0; j < out; j++)
                {
                    int out_cnt = (bat * batch_size + k) * out + j;
                    for (int i = 0; i < in; i++)
                    {
                        int in_cnt = (bat * batch_size + k) * in + i;
                        w[i][j] -= LEARNING_RATE * pre_pardiff[out_cnt] * input[in_cnt];
                    }
                    b[j] -= LEARNING_RATE * pre_pardiff[out_cnt];
                }
            }
            // cout << j << pre_pardiff[j] << ' ';
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
    }

    void send_before(void)
    {
        if (before == NULL)
            return;
        cout << endl << endl << endl;
        for (int bat = 0; bat < len / batch_size; bat++) {
            for (int k = 0; k < batch_size; k++) {
                for (int i = 0; i < in; i++) {
                    for (int j = 0; j < out; j++) {
                        int in_cnt = (bat * batch_size + k) * in + i;
                        int out_cnt = (bat * batch_size + k) * out + j;
                        switch (active)
                        {
                        case sigmoid:
                        case softmax:
                            post_pardiff[in_cnt] += pre_pardiff[out_cnt] * w[i][j] * input[in_cnt] * (1 - input[in_cnt]);
                        case ReLU:
                            if(input[in_cnt]>0)
                                post_pardiff[in_cnt] += pre_pardiff[out_cnt] * w[i][j];
                        }
                    }
                }
            }
        }
        this->before->pre_out = out;
        this->before->pre_pardiff = new double[len * out];
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < out; j++)
            {
                this->before->pre_pardiff[i * out + j] = post_pardiff[i * out + j];
            }
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
    Layer *before, *after;
    Layer(int in_, int out_, int len, enum active_mode active, enum layer_type layer_type)
    {
        // initialization
        srand(time(NULL));
        in = in_;
        out = out_;
        this->len = len;
        this->active = active;
        this->layer_type = layer_type;
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
        post_pardiff = new double[len * out];
        before = NULL;
        after = NULL;
        batch_size = 10;
        if (len < batch_size)
            batch_size = 1;
    }

    void connect(Layer *other)
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
            if (in == 784)
                cout << "training " << i + 1 << endl;
            forwardProp();
            send_after();
            if (after != NULL)
                after->training(1);
            if (layer_type == Output)
            {
                pre_pardiff = new double[len * out];
                for(int j = 0; j < len; j++)
                {
                    for(int k = 0; k < out; k++)
                    {
                        int out_cnt = j * out + k;
                        pre_pardiff[out_cnt] = predict[out_cnt] - output[out_cnt];
                    }
                }
            }
            backwardProp();
            if (layer_type == Output)
                cout << "COST : " << cost() << endl;
                prediction();
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

    // Layer hidden_layer(784, 100, DATA_SET, ReLU, Hidden);
    Layer output_layer(784, 10, DATA_SET, softmax, Output);
    // hidden_layer.connect(&output_layer);

    // hidden_layer.getData(input, output);
    output_layer.getData(input, output);
    output_layer.training(20);
}