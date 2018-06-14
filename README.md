# Machine Learning - Exercise 5

CNNs & Transfer Learning on CIFAR-10 dataset.


## 1. My Neural Network

### Parameters

* **Hidden layer(s):** Two hidden layers in sizes of 100 and 50.
* **Number of epochs:** 10.
* **Learning rate:** 0.001.
* **Architecture:**
    1.  Convolution layer (3 x 50)
    2.  ReLU
    3.  Pool 2D (2 x 2)
    4.  Convolution layer (50 x 16)
    5.  ReLU
    6.  Pool 2D (2 x 2)
    7.  Fully Connected (400 x 100)
    8.  Batch Normalization 1D (100)
    9.  ReLU
    10. Fully Connected (100 x 50)
    11. Batch Normalization 1D (50)
    12. ReLU
    13. Fully Connected (50 x 10)
    14. Softmax
* **Optimizer:** Adam
* **Batch Size:** 128


### Results

* **Training set accuracy:** 78.745%
* **Validation set accuracy:** 67.627%
* **Testing set accuracy:** 68%
* **Average training set loss:** 0.590
* **Average validation set loss:** 0.899
* **Average testing loss sum:** 0.9580

* **Confusion Matrix**:
    ```
    [[586  32  54  23  11   7  13   6 207  61]
     [ 10 794   9   5   1   2   5   1  74  99]
     [ 54  14 558  65  85  79  62  12  52  19]
     [ 14  17  67 554  40 159  72   6  40  31]
     [ 21  12  66  82 639  46  55  29  41   9]
     [ 12   4  51 222  37 577  28  24  25  20]
     [  9   8  30  69  31  33 780   1  31   8]
     [ 13  14  37  60  69 120   7 613  17  50]
     [ 18  28   7   8   2   2   2   0 910  23]
     [ 11  78   7  12   0   4   2   3  83 800]]
    ```
![graph](https://github.com/aedeny/machine_learning-ex5/blob/master/graphs/Training_Loss_vs._Validation_My_Net.png?raw=true)

## 2. Transfer Learning using ResNet-18

### Parameters

* **Number of epochs:** 2.
* **Learning rate:** 0.1.
* **Optimizer:** Adam
* **Batch Size:** 100

### Results

* **Training set accuracy:** 75.782%
* **Validation set accuracy:** 78.440%
* **Testing set accuracy:** 77%
* **Average training set loss:** 0.756
* **Average validation set loss:** 0.006
* **Average testing loss sum:** 0.0067

* **Confusion Matrix**:
    ```
    [[899   7  39   8   1   1   3   6  22  14]
     [ 32 900   6   4   0   1   3   5   5  44]
     [ 43   5 844  33  23   7  29  12   2   2]
     [ 26   7  94 719  14  61  41  27   3   8]
     [ 27   4 163  43 607  17  48  85   3   3]
     [ 12   5  75 176  17 646  22  43   1   3]
     [ 15   6  99  38  14   9 809   7   1   2]
     [ 33   5  58  26  27  15   9 816   1  10]
     [170  39  25  10   1   2   2   5 727  19]
     [ 48  89   6   9   0   0   2   7  12 827]]
    ```