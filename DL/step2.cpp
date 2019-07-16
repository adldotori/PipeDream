// multi variable linear regression
#include <iostream>
#include <limits.h>
#include <float.h>
#define MAX(a,b) (a)>(b)?(a):(b)
#define SQR(a) (a)*(a)
#define LEARNING_RATE 0.00001
#define DATA_CNT 100
#define NUM 3
#define delta 0.00001
#define step 2000
using namespace std;

class Neuron {
private:
    double w[NUM],b, len;
    double input[DATA_CNT][NUM], output[DATA_CNT];

    double feedForward(int data){
        double ret = 0;
        for(int i=0;i<NUM;i++){
            ret += w[i]*input[data][i];
        }
        ret+=b;
        return ret;
    }
    double cost(void){
        // 1. sum of square
        double cost = 0;
        for(int i=0;i<len;i++){
            double val = feedForward(i)-output[i];
            cost += 0.5*SQR(val);
        }
        // etc ...
        return cost;
    }

public:
    Neuron(){
        for(int i=0;i<NUM;i++) w[i]=0;
        b = 0;
    }
    
    void getData(double input[][NUM], double output[],int len){
        this->len = len;
        for(int i=0;i<len;i++){
            for(int j=0;j<NUM;j++)
                this->input[i][j] = input[i][j];
            this->output[i] = output[i];
        }
    }

    void gradientDescent(void){
        double cost_before, cost_after;
        for(int i=0;i<NUM;i++){
            cost_before = cost();
            w[i] += delta;
            cost_after = cost();
            w[i] -= delta;
            w[i] -= LEARNING_RATE * (cost_after - cost_before)/delta;
            // cout<<w[i]<<"d";
        }

        cost_before = cost();
        b += delta;
        cost_after = cost();
        b -= delta;
        b -= LEARNING_RATE * (cost_after - cost_before)/delta;

    } 
    
    void print(void){
        cout << "w =";
        for(int i=0;i<NUM;i++)
            cout << " " << w[i];
        cout << " b = " << b << endl;
        
        cout << "COST : " << cost() << endl;
        
        cout << "Prediction: ";
        for(int i=0;i<len;i++){
            double pred = 0;
            for(int j=0;j<NUM;j++){
                pred += w[j]*input[i][j];
            }
            pred += b;
            cout << pred << " ";
        }
        cout<<endl<<endl;
    }
};

int main(){
    Neuron my_neuron;
    
    double input[DATA_CNT][NUM] = {
        {73,80,75},
        {93,88,93},
        {89,91,90},
        {96,98,100},
        {73,66,70}};
    double output[DATA_CNT] = {152,185,180,196,142};

    my_neuron.getData(input,output,5);
    for(int i=0;i<step;i++){
        my_neuron.gradientDescent();
        if((i+1)%20==0){
            cout<<"training "<<i+1<<endl;
            my_neuron.print();
        }
    }
}
