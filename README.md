# Optical Character Recognition

This OCR model was built using a multi-model deep learning architecture, namely a CNN pretrained on character recognition followed by a bi-directional GRU using [CTC loss](https://distill.pub/2017/ctc/) as the cost function.
The model is able to recognize words in 3 steps. The first step breaks up the image in to overlapping patches accross the width and run them through the CNN which will generate an embedding for each patch. 

![](https://github.com/peterbacalso/ocr/blob/master/assets/demo.gif)

"showcase"

The second step is to then feed these embeddings to the GRU which will output the likelihood of each class for each patch. 
Finally, a greedy decoder takes the most probable character at each timestep, converges duplicates and removes blank labels to form a word prediction.

![](https://github.com/peterbacalso/ocr/blob/master/assets/pipeline.png)

## Data

There are 2 datasets, one is a collection of character images for training the CNN and the other is a collection of word images for training the full model.
All the words used were taken from the [english-words dataset](https://github.com/dwyl/english-words). The character and word images were generated synthetically in 2 ways:

1. Characters from [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) (A handwritten character image dataset). Letter were concatenated to form images for corresponding words.

2. From a selection of 64 fonts, randomly pick a font and generate a letter/word image using [ImageDraw](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html) module from Pillow

![](https://github.com/peterbacalso/ocr/blob/master/assets/sample_words.png)

## CNN Model

1. Conv(32 filters, 7x7 kernel, 1 strides)|BatchNorm|LeakyRelu(.01)|MaxPool(2 pool size)
2. Conv(64 filters, 5x5 kernel, 1 strides)|BatchNorm|LeakyRelu(.01)|MaxPool(2 pool size)
3. Conv(128 filters, 3x3 kernel, 1 strides)|BatchNorm|LeakyRelu(.01)|MaxPool(2 pool size)
4. Conv(256 filters, 3x3 kernel, 1 strides)|BatchNorm|LeakyRelu(.01)|MaxPool(4 pool size)
5. Conv(512 filters, 3x3 kernel, 1 strides)|BatchNorm|LeakyRelu(.01)|Flatten
6. Dropout|Dense (62 units)

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
Epoch 10/10
16/16 [==============================] - 4s 278ms/step - loss: 3798464672.0000 - accuracy: 0.0205 - val_loss: 139443208192.0000 - val_accuracy: 0.0332
```
3. Check that a small learning rate (3e-8) makes the loss go down slowly
```
Epoch 4/5
4/4 [==============================] - 2s 461ms/step - loss: 5.3023 - accuracy: 0.0176 - val_loss: 4.1421 - val_accuracy: 0.0098
Epoch 5/5
4/4 [==============================] - 2s 455ms/step - loss: 5.2544 - accuracy: 0.0156 - val_loss: 4.1415 - val_accuracy: 0.0098
```
4. Using a proper model for training, it should be able to overfit on a small portion of the data. In this case, a subsample of 2000 training images was used.
```
Epoch 199/200
16/16 [==============================] - 4s 279ms/step - loss: 0.1114 - accuracy: 0.9595 - val_loss: 0.8425 - val_accuracy: 0.7852
Epoch 200/200
16/16 [==============================] - 5s 292ms/step - loss: 0.1052 - accuracy: 0.9668 - val_loss: 0.9785 - val_accuracy: 0.7656
```

## CRNN Model

1. Lambda (extract image patches of size 32x16)
2. Lambda (expand image patches to 32x32)
3. CNN (as defined above; already trained)
4. Bi-directional GRU (256 units * 2)
5. Dense (63 units)

### Sanity Check

1. The model should be able to overfit on a single word image. In this case, 5 example words were chosen at random ('midbrains', 'Aurlie', 'proffers', 'cerebrogalactose', 'untapering')

![](https://github.com/peterbacalso/ocr/blob/master/assets/overfit_words.png)

The metric for evaluation is called Levenshtein/edit distance. The value of the metric represents how many edits to the word are needed to match the ground truth word. 
In the results below, a levenshtein metric of 0 means that the prediction and the word match exactly.

![](https://github.com/peterbacalso/ocr/blob/master/assets/overfit.png)

## Results

The crnn used ~29k word images for training. The final validation CTC loss was 2.861 and the validation levenshtein metric was 0.5151.

![](https://github.com/peterbacalso/ocr/blob/master/assets/val_metrics.png)

As for the test set, the Levenshtein/edit distance was 0.4885.
```
143/143 [==============================] - 38s 262ms/step - loss: 1.9271 - levenshtein_metric: 0.4885
```
This means that ~49% of the predictions for the words in the test set were off by one letter.


