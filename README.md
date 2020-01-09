Sanity Check:

When using softmax, the value of the loss when the weights are small and no regularization is used can be approximated by -ln(1/C) = ln(C) where C is the number of classes.

The entire dataset has 62 classes which means the softmax loss should be approximately ln(62)=4.127. After running one forward pass on a neural net with 1 hidden layer, the loss was as expected.
```
1/1 [==============================] - 3s 3s/step - loss: 4.1275 - accuracy: 0.0000e+00 - val_loss: 4.1572 - val_accuracy: 0.0000e+00
```

Adding regularization should make the loss go up. The following test adds l2 regularization of magnitude 1e3 which made the loss jump from 4.127 to 4.3212.
```
1/1 [==============================] - 4s 4s/step - loss: 4.3212 - accuracy: 0.0000e+00 - val_loss: 78.9385 - val_accuracy: 0.0000e+00
```

Next step was to select a small subset of data (20 samples) and get the model to overfit.
```
Epoch 200/200
1/1 [==============================] - 1s 798ms/step - loss: 0.7305 - accuracy: 1.0000 - val_loss: 5.9475 - val_accuracy: 0.1000
```