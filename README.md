# Algomorphism

## Description
General driven framework for object-oriented programming on Artificial Intelligence (AI)

## MetricLossBase architecture:

Base of (Metric and Loss)-Base classes. This class uses 3 types for examples (input or true output) where going through
Metric object (this type of object have a usage of performance of Neural Network). In this case the status is 2. For
Loss object: this type of object have a usage of calculate the gradient of Cost/Loss. In this case the status
is 1. Finally, for input examples, the status is 0. The reason of tow type of status (1 & 2) is that in some cases such as
Zero Shot learning the top output of Neural Network could be a vector where this vector goes through post process into classifier.
In this case only use loss type for top output of Neural Network and input status.
- 0: input
- 1: output loss
- 2: output metric

status examples:

1.
  ```python
status = [
  # input type #1
  [0],
  # output type #1
  [1, 2]
]
  ``` 
  In this status example appears one type of input (this means, the first element of dataset is input). The output type 
  appears one (this mean, the seccond elemnt of dataset is output and measure the cost/loss (code: 1) & metric (code: 2))

2.
  ```python
status = [
  # input type #1
  [0],
  # input type #2
  [0],
  # output type #1
  [1, 2]
]
  ```
  In this status example appears 2 types of input (this means, the first and second element of dataset are inputs). The 
  output type appears one (this mean, the second element of dataset is output and measure the cost/loss (code: 1) & 
  metric (code: 2))

3.   
  ```python
status = [
  # 1st element
  # input type #1
  [0],
  # 2nd element
  # output type #1
  [1],
  # 3rd element
  # output type #2
  [2]
  # 4th element
  # output type #3
  [1, 2]
]
  ```
  In this status example appears one type of input (this means, the first element of dataset is input). The 
  output type appears three times (this mean, the second element of dataset is output and measure the cost/loss (code: 1), 
  the third element is output and measure the metric (code: 2) and the forth element is output and measure cost/loss & metric).
  
## Object example

```python
import algomorphism as am
import tensorflow as tf



class FeedForward(tf.Module, am.base.BaseNeuralNetwork):
    def __init__(self):
        tf.Module.__init__(self, name='feedforward')
        
        status = [
          [0],
          [1, 2]
        ]
        
        am.base.BaseNeuralNetwork.__init__(status)
        
        self.fc1 = am.layers.FC(4, 16)
        self.fc2 = am.layers.FC(16, 32)
        self.out = am.layers.FC(32, 10, 'softmax')
    
    def __call__(self, inputs):
      x = self.fc1(inputs[0])
      x = self.fc2(x)
      y = self.out(x)
      
      return tuple((y,))
```
