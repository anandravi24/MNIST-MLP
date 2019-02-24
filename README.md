# MNIST classifier using MLP # 
This code will help you understand the relationship between hyperparameters and model learning. User can adjust the number of hidden layers, nuerons in the hidden layers, number of epochs and the learning rate and see the effects on the validation accuracy of the dataset mentioned below 

*Usage instructions* <br>

1)Download the Mnist Dataset from https://drive.google.com/file/d/1n3o__3FxAkwmTJZPg0Dl9mkMGlVPLDUH/view?usp=sharing and copy in the same folder with the python Code.

2)The user needs to enter 4 arguments. The arguments required are:
<pre>
  Network_Configuration   Enter the number of neurons in hidden each layer astuple {Seperated by comma} <br>
  Epochs                  Enter the number of epochs <br>
  Minibatch               Enter the minibatch size<br>
  Learning_Rate           Enter the learning_rate<br>
  
 </pre>
  *Sample code*
<pre>  
 In the terminal run the following line:

 $ python3 MNIST-MLP.py 200,100 50 100 0.05
 
 Here,<br> Network_Configuration : 200,100 [200 neurons in hidden layer and 100 neurons in the second hidden layer]<br>
      Epochs: 50<br>
      Minibatch : 100<br>
      Learning Rate: 0.05<br>
 </pre>
 
  
  
  




 
 
 
