# Object Character Recognition

## CNN Model

### Sanity Checks

This step was done to help monitor training and adjust hyperparameters to get good learning results.

1. When using softmax, the value of the loss when the weights are small and no regularization is used can be approximated by -ln(1/C) = ln(C) where C is the number of classes.

The entire dataset has 62 classes which means the softmax loss should be approximately ln(62)=4.127. After running one epoch on a neural net with 1 hidden layer, the loss did in fact match.
```
4/4 [==============================] - 3s 689ms/step - loss: 4.1272 - accuracy: 0.0098 - val_loss: 4.1253 - val_accuracy: 0.0371
```
2. Adding regularization should make the loss go up. The following test adds l2 regularization of magnitude 1e2 which made the loss jump from 0.693 to 2.9.
```
3/3
 [==============================] - 1s 322ms/step - loss: 2.9040 - accuracy: 0.4375 - val_loss: 2.9195 - val_accuracy: 0.6875
```
3. Check that inflating the learning rate to a large value (3e4) makes the loss diverge
```
Epoch 8/10
16/16 [==============================] - 5s 283ms/step - loss: 3611794576.0000 - accuracy: 0.0146 - val_loss: 372927184896.0000 - val_accuracy: 0.0195
Epoch 9/10
16/16 [==============================] - 5s 297ms/step - loss: 4123093296.0000 - accuracy: 0.0156 - val_loss: 867232202752.0000 - val_accuracy: 0.0352
Epoch 10/10
16/16 [==============================] - 4s 278ms/step - loss: 3798464672.0000 - accuracy: 0.0205 - val_loss: 139443208192.0000 - val_accuracy: 0.0332
```
3. Check that a small learning rate (3e-8) makes the loss go down slowly
```
Epoch 2/5
4/4 [==============================] - 2s 575ms/step - loss: 5.3940 - accuracy: 0.0059 - val_loss: 4.1347 - val_accuracy: 0.0137
Epoch 3/5
4/4 [==============================] - 2s 484ms/step - loss: 5.3685 - accuracy: 0.0137 - val_loss: 4.1349 - val_accuracy: 0.0117
Epoch 4/5
4/4 [==============================] - 2s 461ms/step - loss: 5.3023 - accuracy: 0.0176 - val_loss: 4.1421 - val_accuracy: 0.0098
Epoch 5/5
4/4 [==============================] - 2s 455ms/step - loss: 5.2544 - accuracy: 0.0156 - val_loss: 4.1415 - val_accuracy: 0.0098
```
4. Using a proper model for training, it should be able to overfit on a small portion of the data. In this case, a subsample of 2000 training images was used.
```
Epoch 197/200
16/16 [==============================] - 5s 297ms/step - loss: 0.1117 - accuracy: 0.9595 - val_loss: 0.8263 - val_accuracy: 0.7812
Epoch 198/200
16/16 [==============================] - 4s 274ms/step - loss: 0.1114 - accuracy: 0.9590 - val_loss: 0.7888 - val_accuracy: 0.8066
Epoch 199/200
16/16 [==============================] - 4s 279ms/step - loss: 0.1114 - accuracy: 0.9595 - val_loss: 0.8425 - val_accuracy: 0.7852
Epoch 200/200
16/16 [==============================] - 5s 292ms/step - loss: 0.1052 - accuracy: 0.9668 - val_loss: 0.9785 - val_accuracy: 0.7656
```

## RNN Model