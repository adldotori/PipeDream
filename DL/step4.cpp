// mnist(basic neural network)
#include <iostream>
#include <algorithm>
#include <limits.h>
#include <float.h>
#include <math.h>
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

#define MAX(a, b) (a) > (b) ? (a) : (b)
#define SQR(a) (a) * (a)
#define LEARNING_RATE 0.01
#define delta 0.00001
#define DATA_SET 60000
#define active 3       // 1:sigmoid, 2:ReLU
#define how_cost 2     // 1:MSE, 2:ACE
#define how_optimize 2 // 1:gradient descent
using namespace std;

enum mode {train=0,test};

class Network
{
private:
    int in, out, len; // input node, output node, cnt of data
    int batch_size;
    double **w, *b;
    double *input, *output; // data*in, data*out
    double **predict;

    double *activation(double *val)
    {
        double *ret = new double[out];
#if active == 1 // sigmoid
        for (int i = 0; i < out; i++)
            ret[i] = val[i] / (1 + exp(-val[i]));
#elif active == 2 // ReLU
        for (int i = 0; i < out; i++)
        {
            if (val[i] < 0)
                ret[i] = 0;
            else
                ret[i] = val[i];
        }
#elif active == 3 // softmax
        double sum = 0;
        for (int i = 0; i < out; i++)
        {
            sum += exp(val[i]);
        }
        for (int i = 0; i < out; i++)
        {
            ret[i] = exp(val[i]) / sum;
            if(ret[i] < exp(-30)) ret[i] = exp(-30);
        }
#endif
        return ret;
    }

    void feedForward(int data)
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
        predict[data] = activation(ret);
    }

    double getCost(enum mode mode, int batch)
    {
        double cost = 0;
#if how_cost == 1 // MSE(mean square error)
        for (int i = 0; i < len; i++)
        {
            feedForward(i);
            for (int j = 0; j < out; j++)
            {
                double val = predict[i][j] - output[i * out + j];
                cost += 0.5 * SQR(val);
            }
        }
        cost /= len;
#elif how_cost == 2 // ACE(average cross entropy)
        for (int i = batch*batch_size; i < (batch+1)*batch_size; i++)
        {
            if(mode==train)
                feedForward(i);
            for (int j = 0; j < out; j++) {
                cost -= output[i * out + j] * log(predict[i][j]);
            }
        }
        cost /= len;
#endif
        // etc ...
        return cost;
    }

    double optimize(void)
    {
#if how_optimize == 1 // gradient descent
        double cost_before, cost_after;
        for (int j = 0; j < out; j++)
        {
            for (int i = 0; i < in; i++)
            {
                cost_before = cost();
                w[i][j] += delta;
                cost_after = cost();
                w[i][j] -= delta;
                w[i][j] -= LEARNING_RATE * (cost_after - cost_before) / delta;
            }
            cost_before = cost();
            b[j] += delta;
            cost_after = cost();
            b[j] -= delta;
            b[j] -= LEARNING_RATE * (cost_after - cost_before) / delta;
        }
        return cost();
#elif how_optimize == 2 // get cross entropy's derivate function
        double cost = 0;
        for (int bat = 0; bat < len/batch_size; bat++)
        {
            cost += getCost(train, bat);
            for (int j = 0; j < out; j++)
            {
                for(int k = 0; k < batch_size; k++)
                {
                    for (int i = 0; i < in; i++)
                    {
                        w[i][j] -= LEARNING_RATE * (predict[bat*batch_size+k][j]-output[(bat*batch_size+k)*out+j]) * input[(bat*batch_size+k)*in+i];
                    }
                    b[j] -= LEARNING_RATE * (predict[bat*batch_size+k][j]-output[(bat*batch_size+k)*out+j]);
                }
            }
        }
        return cost;
#endif
    }

    void prediction(void)
    {
        int correct = 0;
        for (int i = 0; i < len; i++)
        {
            int max = max_element(predict[i],predict[i]+out)-predict[i];
            if(output[i*out+max]==1) correct++;
        }
        cout << "Correct Rate : " << (double)correct/len << endl << endl;
    }

public:
    Network(int in_, int out_, int len)
    {
        in = in_;
        out = out_;
        batch_size = 1;
        this->len = len;
        w = new double *[in];
        for (int i = 0; i < in; i++)
        {
            w[i] = new double[out];
            for (int j = 0; j < out; j++)
                w[i][j] = 0;
        }
        b = new double[out];
        for (int i = 0; i < out; i++)
            b[i] = 0;
        predict = new double *[len];
        for (int i = 0; i < len; i++)
            predict[i] = new double[out];
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
            cout << "cost : " << optimize() << endl;
        }
        prediction();
    }
};

void download(double * input[], double * output[])
{
    unsigned int cnt;
    mnist_data *data;
    int ret = mnist_load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &data, &cnt);
    if (ret) {
        cout << "An error occured: " << ret << endl;
    }
    else {
        cout << "image count: " << cnt << endl;
    }
    int tmp=0;
    for (int i = 0; i < DATA_SET; i++) {
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 28; k++) {
                *(*input+tmp++) = data[i].data[j][k];
            }
        }
        *(*output+i*10+data[i].label) = 1;
    }
    return;
}

int main()
{
    double * input = new double[784*DATA_SET];
    double * output = new double[784*DATA_SET];
    download(&input,&output);
    Network net(784, 10, DATA_SET);
    
    net.getData(input, output);
    net.training(50);
}