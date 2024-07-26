'''
\n Neural Network by Wesley :D
\n No Tensorflow, Keras, Pytorch, etc...        
\n Just numpy and pickle (for saving weights) :D
'''
import numpy as np
import pickle
class NeuralNetwork:
   def __init__(self,n_input):
      self.n_input = n_input
      self.weights = []
      self.biases = []
      self.n_hidden = 0
      self.hidden_neuron = []
      self.hidden_activation = []
   def add_hidden(self,neuron,activation):
      self.hidden_neuron.append(neuron)
      self.hidden_activation.append(activation)
      if(self.n_hidden==0):self.weights.append(np.random.uniform(-0.5,0.5,(neuron,self.n_input)))
      else:self.weights.append(np.random.uniform(-0.5,0.5,(neuron,self.hidden_neuron[self.n_hidden-1])))
      self.biases.append(np.ones(neuron))
      self.n_hidden+=1
   def sigmoid(self,x):return 1/(1+np.exp(-x))
   def relu(self,x):return np.maximum(0,x)
   def sigmoid_derivative(self, x):return x*(1-x)
   def relu_derivative(self, x):return np.where(x>0,1,0)
   def add_output(self,neuron,activation):
      self.hidden_neuron.append(neuron)
      self.hidden_activation.append(activation)
      if(self.n_hidden==0):self.weights.append(np.random.uniform(-0.5,0.5,(neuron,self.n_input)))
      else:self.weights.append(np.random.uniform(-0.5,0.5,(neuron,self.hidden_neuron[self.n_hidden-1])))
      self.biases.append(np.ones(neuron))
   def forward(self,image):
      self.outputs = [image]
      output = image
      for i in range(self.n_hidden):
         z = np.dot(self.weights[i],output) + self.biases[i]
         if self.hidden_activation[i]=='sigmoid':
            output = self.sigmoid(z)
         elif self.hidden_activation[i]=='relu':
            output = self.relu(z)
         self.outputs.append(output)
      z = np.dot(self.weights[self.n_hidden],output) + self.biases[self.n_hidden]
      if self.hidden_activation[i]=='sigmoid':
         output = self.sigmoid(z)
      elif self.hidden_activation[i]=='relu':
         output = self.relu(z)
      self.outputs.append(output)
      return output
   def backward(self,learning_rate, target):
      delta_o = self.outputs[self.n_hidden+1]-target
      self.weights[self.n_hidden]-=learning_rate * np.dot(delta_o.reshape(-1,1),self.outputs[self.n_hidden].reshape(1,-1))
      self.biases[self.n_hidden]-=learning_rate * delta_o
      prev_delta = delta_o
      for i in range(self.n_hidden-1,-1,-1):
         if self.hidden_activation[i]=='sigmoid':
            delta_h = np.dot(self.weights[i+1].T, prev_delta) * self.sigmoid_derivative(self.outputs[i+1])
         elif self.hidden_activation[i]=='relu':
            delta_h = np.dot(self.weights[i+1].T, prev_delta) * self.relu_derivative(self.outputs[i+1])
         self.weights[i]-=learning_rate * np.dot(delta_h.reshape(-1,1),self.outputs[i].reshape(1,-1))
         self.biases[i]-=learning_rate * delta_h
         prev_delta = delta_h
   def train(self,data,targets,epochs,learning_rate,return_value=[],print_output=True):
      best_accuracy = accuracy = 0
      for epoch in range(epochs):
         correct = 0
         for image,target in zip(data,targets):
            output = self.forward(image)
            self.backward(learning_rate, target)
            if(np.argmax(output)==np.argmax(target)):correct+= 1
         accuracy = correct/len(data)
         if print_output:print('Epoch %d/%d, Accuracy: %.2f%%'%(epoch+1, epochs, accuracy*100))
         best_accuracy = max(accuracy,best_accuracy)
      results = {}
      for item in return_value:
         if item=='accuracy':results['accuracy'] = best_accuracy*100
         elif item=='weights':results['weights'] = self.weights
         elif item == 'biases':results['biases'] = self.biases
      return results
   def save_weights(self, filepath):
      with open(filepath, 'wb') as f:
         pickle.dump((self.weights, self.biases), f)
      print(f"Weights and biases saved to {filepath}")
   def load_weights(self, filepath):
      with open(filepath, 'rb') as f:
         self.weights, self.biases = pickle.load(f)
      print(f"Weights and biases loaded from {filepath}")
   def predict(self,image):
      output = self.forward(image)
      return output