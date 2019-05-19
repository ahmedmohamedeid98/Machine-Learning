import numpy
import scipy.special
import imageio
from Gui import Ui_Form



############################################################################
#                               network                                    #
############################################################################
gui_object = Ui_Form()
class HandWrittinRecognition:
    
    
    def __init__(self, input_vector, inputLayer, outputLayer, learningRate):
        # set number of nodes in each input, hidden, output layer
        self.input_vector = input_vector  #784 = 28*28 image size
        self.input_layer = inputLayer  #200 neurons
        self.output_layer = outputLayer # 10 neurons
        self.accuracy_str = ""
        training_data_file = open("train_data.csv", 'r')
        self.training_data = training_data_file.readlines()
        training_data_file.close()
                                                         # range                           # dimention
        self.weight_input = numpy.random.normal(0.0, pow(self.input_vector, -0.5), (self.input_layer, input_vector)) # [200 x 784]
        self.weight_output = numpy.random.normal(0.0, pow(self.input_layer, -0.5), (self.output_layer, self.input_layer)) # [10x200]

        # learning rate
        self.lr = learningRate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x) 
        
        pass

    def Update_Weight(self, inputs_list, targets_list):
        # convert inputs list to Vector
        inputs = numpy.array(inputs_list, ndmin=2).T  # vector [784 x 1]
        targets = numpy.array(targets_list, ndmin=2).T # vector [10 x 1]
        
        # net_input_layer = w_input . p(inputs)
        net_input_layer = numpy.dot(self.weight_input, inputs)
        # output = f(net_input_layer)
        input_layer_outputs = self.activation_function(net_input_layer)
        
        net_output_layer = numpy.dot(self.weight_output, input_layer_outputs)
        output_layer_outputs = self.activation_function(net_output_layer)
        
        
        #####         Update Weight           ########
        
        # output layer error : (target - actual)
        output_layer_errors = targets - output_layer_outputs
        # hidden layer error
        input_layer_errors = numpy.dot(self.weight_output.T, output_layer_errors)
        
        # update the weights between the input_layer and output_layer
        self.weight_output += self.lr * numpy.dot((output_layer_errors * output_layer_outputs * (1.0 - output_layer_outputs)), numpy.transpose(input_layer_outputs))
        
        # update the weights between the input_vector and input_layer
        self.weight_input += self.lr * numpy.dot((input_layer_errors * input_layer_outputs * (1.0 - input_layer_outputs)), numpy.transpose(inputs))
        
        pass

   
    def test(self, inputs_list):
        # convert list into vector
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # N = W.P
        net_input_layer = numpy.dot(self.weight_input, inputs)
        # output = f(N)
        input_layer_outputs = self.activation_function(net_input_layer)
        
        
        net_output_layer = numpy.dot(self.weight_output, input_layer_outputs)
        output_layer_outputs = self.activation_function(net_output_layer)
        
        return output_layer_outputs
    
    def get_output(self, img_data):
        output =  self.test(img_data)
        return output
    
    
    def test_preformance(self,accuracy_str): 
        test_data_file = open("test_data.csv", 'r') # 1000
        test_data_list = test_data_file.readlines()
        test_data_file.close()
        scorecard = []
        for record in test_data_list:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # normalization 0.01 <--> 1
            
            outputs = self.test(inputs) # Outputs --> vector [10x1]
            label = numpy.argmax(outputs)
            if (label == correct_label):
                scorecard.append(1)
            else:
                scorecard.append(0)
                pass

            pass
        scorecard_array = numpy.asarray(scorecard)
        
        accuracy_str += str( scorecard_array.sum() / scorecard_array.size )[:-4]
        accuracy_str += '\n'
        return accuracy_str
    
    
    def train_model(self):        
        
        for record in self.training_data:
            
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(self.output_layer) + 0.01
            targets[int(all_values[0])] = 0.99
            self.Update_Weight(inputs, targets)
            pass
        self.accuracy_str = self.test_preformance(self.accuracy_str)
        return  self.accuracy_str
        
