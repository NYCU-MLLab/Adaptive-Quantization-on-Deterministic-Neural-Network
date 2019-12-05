## Adaptive quantization neural network
Adaptive quantization neural network is a model that generalizes quantization models from binary to 'M'-ary. The representation values and quantization partitions are adaptively updated by end-to-end method. After the training, one might obtain an asymmetric quantizer which saves memory storage but conserves the classified accuracy.
## File description
"main_nonBayes.py" is the main file that runs adaptive quantization neural network.

"main_nonBayesbackup.py" is the main file that runs full precision model.

"utils" contains quantizaiton models and vanilla models.
## Result
After the adaptive quantization method, the weights will prone to the representation values. In the picture below, the blue sticks are the histogram of weights and the black dashed lines are the representation values in the layer.
![image](https://github.com/susan0720/Su-Ting-Chang-Adaptive-Quantization-on-Deterministic-Neural-Network/blob/master/nonBayesM2.png)
## Setting
* Hardware:
  * CPU: Intel Core i7-4930K @3.40 GHz
  * RAM: 64 GB DDR3-1600
  * GPU: GeForce GTX 1080ti
* pytorch 
* Dataset
  * MNIST
  * CIFAR10
