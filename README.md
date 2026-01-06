# Machine-Learning-Neural-Networks-Project

## Files in this project:
1. models.py : Perceptron and Neural Network models for a variety of applications
   - Perceptron
     - PerceptronModel class in models.py
   - Non-linear regression
     - RegressionModel class in models.py  
3. nn.py : Neural network mini-library
4. backend.py : Backend code for various machine learning tasks
5. data : Datasets for digit classification and language identification

## Digit Classification
- DigitClassificationModel class in models.py
- Neural Network to classify handwritten digits from MNIST dataset
- Each digit is of size 28 by 28 pixels, the values of which are stored in a 784-dimensional vector of floating point numbers
- Each output is a 10-dimensional vector which has zeros in all positions, except for a one in the position corresponding to the correct class of the digit
- Run DigitClassificationModel.run() to return a batch_size by 10 node containing scores, where higher scores indicate a higher probability of a digit belonging to a particular class
- Use dataset.get_validation_accuracy() to compute validation accuracy for model using validation data
- To test the implementation, run the following: python autograder.py -q q3 

## Language Identification
- LanguageIDModel class in models.py
- Neural Network that identifies the language for one word at a time
- Dataset contains words in different languages
- To test the implementation, run the following: python autograder.py -q q4
