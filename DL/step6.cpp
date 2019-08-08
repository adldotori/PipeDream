// mnist(advanced multi layer neural network)
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
#define beta1 0.9
#define beta2 0.999
#define epsilon 0.00000001
#define optimize 2
#define LEARNING_RATE 0.001
#define DATA_SET 1000
#define TEST_DATA_SET 10000
#define BATCH_SIZE 1
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
    double *m, *v, *b_m, *b_v;          // adam variable
    double *input, *output; // data*in, data*out
    double *predict;
    double *pre_pardiff;
    Layer *before, *after;
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
                if(val[i] > 300) cout << val[i] << ' ';
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
                ret[i] = 0;
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

    void backwardProp(int batch, int step)
    {
#if optimize == 1 // Stochastic Gradient Descent(SGD)
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
#elif optimize == 2 // Adaptive Moment Estimation(Adam)
        for (int k = batch * batch_size; k < (batch + 1) * batch_size; k++)
        {
            int step_cnt = step * len + k + 1;
            for (int j = 0; j < out; j++)
            {
                int out_cnt = k * out + j;
                for (int i = 0; i < in; i++)
                {
                    int in_cnt = k * in + i;
                    int mv_cnt = i * out + j;
                    // if(mv_cnt == 123) cout << pre_pardiff[out_cnt] << ' ' << input[in_cnt] << endl;
                    m[mv_cnt] = beta1 * m[mv_cnt] + (1 - beta1) * pre_pardiff[out_cnt] * input[in_cnt];
                    v[mv_cnt] = beta2 * v[mv_cnt] + (1 - beta2) * pre_pardiff[out_cnt] * pre_pardiff[out_cnt] * input[in_cnt] * input[in_cnt];
                    // if(i == 510) cout << j << ' ' <<step_cnt << ' ' << pre_pardiff[out_cnt] * pre_pardiff[out_cnt] * input[in_cnt] * input[in_cnt] << endl;
                    // if(mv_cnt == 123) cout << m[mv_cnt] << ' ' << v[mv_cnt] << ' ' << (LEARNING_RATE * sqrt(1-pow(beta2, step_cnt)) / (1-pow(beta1, step_cnt))) * m[mv_cnt] / (sqrt(v[mv_cnt]) + epsilon) << endl;
                    if((LEARNING_RATE * sqrt(1-pow(beta2, step_cnt)) / (1-pow(beta1, step_cnt))) * m[mv_cnt] / (sqrt(v[mv_cnt]) + epsilon)>1)cout << step_cnt << ' ' << i << ' '<<j<<' '<<(LEARNING_RATE * sqrt(1-pow(beta2, step_cnt)) / (1-pow(beta1, step_cnt))) * m[mv_cnt] / (sqrt(v[mv_cnt]) + epsilon) << ' ' << m[mv_cnt] << ' ' << v[mv_cnt] << endl;
                    w[i][j] -= LEARNING_RATE * (sqrt(1-pow(beta2, step_cnt)) / (1-pow(beta1, step_cnt))) * m[mv_cnt] / (sqrt(v[mv_cnt] + epsilon));
                }
                b_m[j] = beta1 * b_m[j] + (1 - beta1) * pre_pardiff[out_cnt];
                b_v[j] = beta2 * b_v[j] + (1 - beta2) * pre_pardiff[out_cnt] * pre_pardiff[out_cnt];
                b[j] -= LEARNING_RATE * (sqrt(1-pow(beta2, step_cnt)) / (1-pow(beta1, step_cnt))) * b_m[j] / (sqrt(b_v[j]) + epsilon);
            }
        }
#endif
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

    void send_after(int batch)
    {
        if (after == NULL)
            return;
    }

    void send_before(int batch)
    {
        if (before == NULL)
            return;

        double *post_pardiff = new double[len * in];
        for(int i = 0; i < len * in; i++)
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
                    int out_cnt = k * out + j;
                    switch (active)
                    {
                    case sigmoid:
                    case softmax:
                        post_pardiff[in_cnt] += pre_pardiff[out_cnt] * w[i][j] * input[in_cnt] * (1 - input[in_cnt]);
                        break;
                    case ReLU:
                        if (input[in_cnt] > 0)
                            post_pardiff[in_cnt] += pre_pardiff[out_cnt] * w[i][j];
                        break;
                    }
                }
            }
        }
        if (batch == 0)
        {
            before->pre_pardiff = new double[len * in];
        }
        for (int k = batch * batch_size; k < (batch + 1) * batch_size; k++)
        {
            for (int i = 0; i < in; i++)
            {
                before->pre_pardiff[k * in + i] = post_pardiff[k * in + i];
            }
        }
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
    Layer(int in_, int out_, int len, enum active_mode active, enum layer_type layer_type)
    {
        // initialization
        in = in_;
        out = out_;
        this->len = len;
        this->active = active;
        this->layer_type = layer_type;
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
        m = new double[in * out];
        v = new double[in * out];
        for (int i = 0; i < in * out; i++)
        {
            m[i] = 0;
            v[i] = 0;
        }
        b = new double[out];
        b_m = new double[out];
        b_v = new double[out];
        for (int i = 0; i < out; i++)
        {
            b[i] = 0;
            b_m[i] = 0;
            b_v[i] = 0;
        }
        predict = new double[len * out];
        before = NULL;
        after = NULL;
        batch_size = BATCH_SIZE;
        if (len < batch_size)
            batch_size = 1;
    }

    void connect(Layer *other)
    {
        if (other->in != this->out)
        {
            cout << "It's impossible because the number of layer's node is different." << endl;
            exit(1);
        }
        this->after = other;
        other->before = this;
    }

    void getData(double *input, double *output)
    {
        this->input = input;
        if (after != NULL)
        {
            after->getData(predict, output);
        }
        else
        {
            this->output = output;
        }
    }

    void batch_training(int batch, int step)
    {
        forwardProp(batch);
        send_after(batch);
        if (after != NULL)
            after->batch_training(batch, step);
        if (layer_type == Output)
        {
            pre_pardiff = new double[len * out];
            for (int i = batch * batch_size; i < (batch + 1) * batch_size; i++)
            {
                for (int j = 0; j < out; j++)
                {
                    int out_cnt = i * out + j;
                    pre_pardiff[out_cnt] = predict[out_cnt] - output[out_cnt];
                }
            }
        }
        backwardProp(batch, step);
        send_before(batch);
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
                batch_training(j, i);
            }
        }
    }

    void test(void)
    {
        cout << "TEST" << endl;
        len = TEST_DATA_SET;
        batch_size = len;
        forwardProp(0);
        send_after(0);
        if (after != NULL)
            after->test();
        if (layer_type == Output)
        {
            prediction();
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

int main()
{
    double *input = new double[784 * DATA_SET];
    double *output = new double[10 * DATA_SET];
    double *test_input = new double[784 * TEST_DATA_SET];
    double *test_output = new double[10 * TEST_DATA_SET];
    download(&input, &output);
    download_test(&test_input, &test_output);

    // Layer output_layer(784, 10, DATA_SET, softmax, Output);

    // output_layer.getData(input, output);
    // output_layer.training(30);

    Layer hidden_layer(784, 256, DATA_SET, ReLU, Hidden);
    Layer output_layer(256, 10, DATA_SET, softmax, Output);
    hidden_layer.connect(&output_layer);

    hidden_layer.getData(input, output);
    hidden_layer.training(15);

    // Layer hidden_layer1(784, 256, DATA_SET, ReLU, Hidden);
    // Layer hidden_layer2(256, 256, DATA_SET, ReLU, Hidden);
    // Layer output_layer(256, 10, DATA_SET, softmax, Output);
    // hidden_layer1.connect(&hidden_layer2);
    // hidden_layer2.connect(&output_layer);

    // hidden_layer1.getData(input, output);
    // hidden_layer1.training(30);

    // hidden_layer.getData(test_input, test_output);
    // hidden_layer.test();
}