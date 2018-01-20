from __future__ import division
import math
import random
import pandas as pd
import sys
#**************Authors: Gokul Surendra & Padma Kurdgi*******************************************
class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.output = 0
        self.delta = 0

    def summing_unit(self, input):
        sum = 0
        for i in range(len(input) - 1):
            sum += self.weights[i] * input[i]
        return sum + self.bias

    def neuron_output(self, input):
        x = self.summing_unit(input)
        self.output = self.sigmoid(x)
        return self.output

    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            if x > 0:
                return 1
            else:
                return 0

    def get_error(self, target):    #Mean Square Error
        return (0.5) * (target - self.output) ** 2


class NeuralLayer:
    def __init__(self, no_neuron):
        self.bias = random.uniform(-0.05,0.05)
        self.neurons = []
        for i in range(no_neuron):
            self.neurons.append(Neuron(self.bias))


class NeuralNetwork:
    def __init__(self, num_input, num_hidden_layers ,num_hidden_neuron, num_output):
        self.num_input = num_input
        self.hidden_layers = [NeuralLayer(num_hidden_neuron) for i in range(num_hidden_layers)]
        self.output_layer = NeuralLayer(num_output)
        self.num_output = num_output
        self.init_weights_to_hidden() # initialise random weights
        self.init_weights_hidden_to_hidden()
        self.init_weights_to_output()

    def init_weights_to_hidden(self):
        for i in range(len(self.hidden_layers[0].neurons)):
            for j in range(self.num_input):
                self.hidden_layers[0].neurons[i].weights.append(random.uniform(-0.05,0.05))

    def init_weights_hidden_to_hidden(self):
        for i in range(1, len(self.hidden_layers)):
            for j in range(len(self.hidden_layers[i].neurons)):
                for k in range(len(self.hidden_layers[i-1].neurons)):
                    self.hidden_layers[i].neurons[j].weights.append(random.uniform(-0.05,0.05))

    def init_weights_to_output(self):
        last_hidden = len(self.hidden_layers)-1
        for i in range(self.num_output):
            for j in range(len(self.hidden_layers[last_hidden].neurons)):
                self.output_layer.neurons[i].weights.append(random.uniform(-0.05,0.05))

    def train_network(self, dataset, max_iter, expected_output, y_train):
        k = 0
        dataset = dataset.reset_index(drop= True)
        while k < max_iter:
            for index, row in dataset.iterrows():
                output = self.forward_propagation(list(row))

                target = expected_output[index]

                self.back_propagation(row, target, output)

            k+=1
        print 'Total Training Error & Accuracy:'
        self.test(dataset, y_train)


    def print_model(self):
        j=1
        for layer in self.hidden_layers:
            i=1
            print '\n'+'Hidden Layer:'+ str(j)
            for neuron in layer.neurons:
                print 'Neuron '+str(i)+' weights:'+ str(neuron.weights)
                i+=1
            j+=1
        print '\n'+'Output Layer:'
        k = 1
        for neuron in self.output_layer.neurons:
            print 'Neuron ' + str(k) + ' weights:' + str(neuron.weights)
            k+=1

    def test(self, input, y_train):
        out_pred  = []

        for index, row in input.iterrows():
            output = self.forward_propagation(list(row))
            prediction_val = max(output)
            prediction_ind = output.index(prediction_val)
            out_pred.append(prediction_ind)

        self.total_error(out_pred, y_train)
        self.get_accuracy(out_pred, y_train)


    def get_accuracy(self, classify, actual_classify):
        count = 0
        for entry, actual_entry in zip(classify, actual_classify):
            if entry == actual_entry:
                count += 1
        print (count / len(actual_classify)) * 100


    def total_error(self, prediction, actual):
        error =0
        for entry, actual_entry in zip(prediction, actual):
            error += 0.5 *(actual_entry-entry)**2
        print error/ len(actual)

    def back_propagation(self, entry, target, output):
        # Last output layer deltas
        for i in range(len(self.output_layer.neurons)):
            self.output_layer.neurons[i].delta = (target[i] - output[i]) * (1 - output[i]) * output[i]

        #calculate deltas of Last hidden layer
        for index, neuron in enumerate(self.hidden_layers[-1].neurons):
            prod_sum = 0
            for next in self.output_layer.neurons:
                prod_sum += next.delta * next.weights[index]
            neuron.delta = neuron.output*(1-neuron.output)*prod_sum #(self.output_layer.neurons[0].delta*self.output_layer.neurons[0].weights[index])
        #update weights - output neurons
        for neuron in self.output_layer.neurons:
            for index in range(len(neuron.weights)):
                neuron.weights[index] +=  neuron.delta * self.hidden_layers[-1].neurons[index].output

        #calculate deltas Hidden layer
        for ind in range(len(self.hidden_layers)-2,-1,-1):
            for id, neuron in enumerate(self.hidden_layers[ind].neurons):
                prod_sum = 0
                for next in self.hidden_layers[ind+1].neurons:
                    prod_sum += next.delta * next.weights[id]
                neuron.delta = neuron.output*(1-neuron.output)* prod_sum
        #update weights - hidden neurons
        for ind_layer in range(len(self.hidden_layers)-1,0,-1):
            for neur in self.hidden_layers[ind_layer].neurons:
                for w_index in range(len(neur.weights)):
                    neur.weights[w_index] += neur.delta * self.hidden_layers[ind_layer-1].neurons[w_index].output
        #update weights from input to hidden
        for neuron in self.hidden_layers[0].neurons:
            for index in range(len(neuron.weights)):
                neuron.weights[index] += neuron.delta * entry[index]


    def forward_propagation(self, entry):
        hid_output = entry[:]
        for layer in self.hidden_layers:
            temp = []
            for neuron in layer.neurons:
                temp.append(neuron.neuron_output(hid_output))
            hid_output = temp[:]
        output = []
        for i in range(len(self.output_layer.neurons)):
            output.append(self.output_layer.neurons[i].neuron_output(hid_output))

        return output

def split_data(dataset, training_perc):
    train = dataset.sample(frac = (training_perc/100))
    test = dataset.drop(train.index)
    x_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    x_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    return x_train,y_train,x_test,y_test


def main():
    input_dataset = None; training_perc =None; max_iter = None; num_hidden_layers=None;num_neuron=None
    if len(sys.argv) < 6:
        print "Incorrect no. of arguments"
    else:
        path = sys.argv[1]
        input_dataset = pd.read_csv(path, header =-1)
        training_perc = int(sys.argv[2])
        max_iter = int(sys.argv[3])
        num_hidden_layers = int(sys.argv[4])
        num_neuron = int(sys.argv[5])

    x_train, y_train, x_test, y_test = split_data(input_dataset, training_perc)

    y_train_list = []
    for x in y_train:
        temp= []
        temp.insert(x, 1)
        for i in range(y_train.nunique()):
            if i != x:
                temp.insert(i,0)
        y_train_list.append(temp)

    new_neural = NeuralNetwork(len(x_train.columns), num_hidden_layers, num_neuron, y_train.nunique())
    new_neural.train_network(x_train, max_iter, y_train_list, y_train)

    print '\n'+'Total Test Error & Accuracy:'
    new_neural.test(x_test,y_test)

    new_neural.print_model()

if __name__ == "__main__":
    main()