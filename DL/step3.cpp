// basic neural network
#include <iostream>
#include <limits.h>
#include <float.h>
#include <math.h>
#define MAX(a,b) (a)>(b)?(a):(b)
#define SQR(a) (a)*(a)
#define LEARNING_RATE 0.1
#define delta 0.00001
#define active 3 // 1:sigmoid, 2:ReLU
#define how_cost 2 // 1:MSE, 2:ACE
#define how_optimize 1 // 1:gradient descent
using namespace std;

class Network {
private:
    int in,out,len; // input node, output node, cnt of data
    double ** w,* b;
    double * input, * output; // data*in, data*out
    double ** predict;

    double * activation(double * val){
        double * ret = new double[out];
#if active==1 // sigmoid
        for(int i=0;i<out;i++)
            ret[i] = val[i]/(1+exp(-val[i]));
#elif active==2 // ReLU
        for(int i=0;i<out;i++) {
            if(val[i]<0) ret[i]=0;
            else ret[i]=val[i];
        }
#elif active==3 // softmax
        double sum=0;
        for(int i=0;i<out;i++){
            sum += exp(val[i]);
        }
         for(int i=0;i<out;i++){
            ret[i] = exp(val[i])/sum;
        }
#endif  
        return ret;
    }

    void feedForward(int data){
        double * ret = new double[out];
        for(int i=0;i<out;i++){
            for(int j=0;j<in;j++){
                ret[i] += w[j][i]*input[data*in+j];
            }
            ret[i]+=b[i];
        }
        predict[data] = activation(ret);
    }

    double cost(void){
        double cost = 0;
#if how_cost==1 // MSE(mean square error)
        for(int i=0;i<len;i++){
            feedForward(i);
            for(int j=0;j<out;j++){
                double val = predict[i][j]-output[i*out+j];
                cost += 0.5*SQR(val);
            }
        }
        cost /= len;
#elif how_cost==2 // ACE(average cross entropy)
        for(int i=0;i<len;i++){
            feedForward(i);
            for(int j=0;j<out;j++){
                cost -= output[i*out+j]*log(predict[i][j]);
            }
        }
        cost /= len;
#endif
        // etc ...
        return cost;
    }
    void optimize(void){
#if how_optimize==1 // gradient descent
        double cost_before, cost_after;
        for(int j=0;j<out;j++){
            for(int i=0;i<in;i++){
                cost_before = cost();
                w[i][j] += delta;
                cost_after = cost();
                w[i][j] -= delta;
                w[i][j] -= LEARNING_RATE * (cost_after - cost_before)/delta;
            }
            cost_before = cost();
            b[j] += delta;
            cost_after = cost();
            b[j] -= delta;
            b[j] -= LEARNING_RATE * (cost_after - cost_before)/delta;
        }
#endif
    }     
    
    void print(void){
        cout << "COST : "<<cost()<<endl;
        // cout << "WEIGHT, BIAS"<< endl;
        // for(int i=0;i<out;i++){
        //     cout << i << " : ";
        //     for(int j=0;j<in;j++){
        //         cout<< w[j][i] << " ";
        //     }
        //     cout << " | " << b[i] << endl;
        // }
    }
    void prediction(void){
        cout << endl;
        cout << "Prediction: " << endl;
        for(int i=0;i<len;i++){
            feedForward(i);
            for(int j=0;j<out;j++){
                cout<<predict[i][j]<< " ";
            }
            cout << endl;
        }
        cout<<endl<<endl;
    }
    
public:
    Network(int in_,int out_,int len){
        in = in_;
        out = out_;
        this->len = len;
        w = new double * [in];
        for(int i=0;i<in;i++) {
            w[i] = new double[out];
            for(int j=0;j<out;j++)
                w[i][j]=0;
        }
        b = new double[out];
        for(int i=0;i<out;i++)
            b[i]=0;
        predict = new double * [len];
        for(int i=0;i<len;i++)
            predict[i] = new double[out];
    }
    
    void getData(double * input, double * output){
        this->input = input;
        this->output = output;
    }

    void training(int step){
        for(int i=0;i<step;i++){
            optimize();
            if((i+1)%50==0){
                cout<<"training "<<i+1<<endl;
                print();
            }
        }
        prediction();
    }
};

int main(){
    Network net(4,3,8);
    double input[] = {
        1,2,1,1,
        2,1,3,2,
        3,1,3,4,
        4,1,5,5,
        1,7,5,5,
        1,2,5,6,
        1,6,6,6,
        1,7,7,7};
    double output[] = {
        0,0,1,
        0,0,1,
        0,0,1,
        0,1,0,
        0,1,0,
        0,1,0,
        1,0,0,
        1,0,0};
    net.getData(input,output);
    net.training(2000);
}