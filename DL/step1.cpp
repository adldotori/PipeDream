// linear regression
#include <iostream>
#include <limits.h>
#include <float.h>
#define MAX(a,b) (a)>(b)?(a):(b)
#define SQR(a) (a)*(a)
#define LEARNING_RATE 0.01
#define DATA_CNT 100
#define delta 0.00000001
#define step 2000
using namespace std;

class Neuron {
private:
    double w,b, len;
    double input[DATA_CNT], output[DATA_CNT];

public:
    Neuron():w(0.0),b(0.0){}
    Neuron(const double& w_input, const double& b_input):w(w_input),b(b_input){}
    
    void getData(double * input, double * output,int len){
        this->len = len;
        for(int i=0;i<len;i++){
            this->input[i] = input[i];
            this->output[i] = output[i];
        }
    }
    double feedForward(const double& input){
        return w*input + b;
    }
    double cost(void){
        // 1. sum of square
        double cost = 0;
        for(int i=0;i<len;i++){
            cost += 0.5*SQR(feedForward(input[i])-output[i]);
        }
        // etc ...
        return cost;
    }

    void gradientDescent(void){
        double cost_before = cost();
        w += delta;
        double cost_after = cost();
        w -= delta;
        w -= LEARNING_RATE * (cost_after - cost_before)/delta;

        cost_before = cost();
        b += delta;
        cost_after = cost();
        b -= delta;
        b -= LEARNING_RATE * (cost_after - cost_before)/delta;
    } 
    
    void print(void){
        cout << "w = " << w << " b = " << b << endl;
    }
};

int main(){
    Neuron my_neuron;
    
    double input[] = {1,2,3};
    double output[] = {1,2,3};

    my_neuron.getData(input,output,3);
    for(int i=0;i<step;i++){
        my_neuron.gradientDescent();
        if((i+1)%20 == 0) {
            cout<<"training "<<i+1<<endl;
            my_neuron.print();
        }
    }
}