## Experiment Logs

### 6-Class classification of rotten or fresh apples, oranges, and bananas
Experiment setup:
- Initial learning rate 0.0001
- Default mini-batch size 10
- Default epoch size 6
- investigate the effect of varying mini-batch size (5, 10, 50, 100)
- investigate the number of epochs (3, 6)


varying epoch size:
- Inference without training: 18.1%
- Inference after training 1 epoch: 90.9%
- Inference after training 3 epoch: 95.4%
- Inference after training 6 epoch: 98.6%

Varying mini-batch size:
- batch size 5: 98.8%
- batch size 10: 98.6%
- batch size 50: 94.2%
- batch size 100: 92.2%

After applying the preprocessing on both training and validation data, the training validation accuracy of default setup is 96.7%,
and the testing accuracy is 96.4%. 

But if I don't apply preprocessing, the testing accurary is 98.0%

### Binary classification of rotten fruit or vegetables

For the upgraded dataset, for each kind of fruit or vegetables, we perform a train loop and a validation, 
with batch size 10 and epoch size 10.

The validation accuracy of each category:

- Apple: 97.8%
- Banana: 99.7%
- Mango: 99.7%
- Orange: 97.8%
- Strawberry: 99.2%
- Bell pepper: 96.1%
- Carrot: 95.3%
- Cucumber: 98.6%
- Potato: 95.3%
- Tomato: 99.4%

### Rotten Food Detection of 5 kinds of fruits and 5 kinds of vegetables

In this experiment, we perform a 20-class classification of the upgraded dataset, which contains 5 types of fruits and 
5 types of vegetables, each kind of which will be predicted as rotten *something* or fresh *something*. 

As default, we set the batch size as 10 and the epoch size as 6. The validation accuracy is 95.6%.

### General Sense Rotten Food Detection.

In this experiment, we only predict two classes, `rotten` and `fresh`, and we perform four experiments.

First, we perform classification on the original dataset, the validation accuracy is 98.6%.

Second, we perform classification on the upgraded dataset, the validation accuracy is 97.7%.

Third, we train on the upgraded dataset and test on the original dataset without Zicong's refined preprocessing methods. 
The test accuracy without preprocessing is 91.5%, and accuracy with preprocessing is 92.8%

Last, we train on the original dataset and test on the upgraded dataset with and without Zicong's refined preprocessing methods. 
The test accuracy without preprocessing in the training part is 79.7, the test accuracy with preprocessing is 77.8%

