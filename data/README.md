# Data engine for datasets

## [Fruits fresh and rotten for classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)

Which is the original dataset used in the paper. To reimplement the algorithm proposed in the paper, we create the following 
dataset class for loading the images.

- 6-Class classification of rotten or fresh apples, oranges, and bananas (_**all_fruit.py**_)

where 350 and 150 images are selected as the training and validation dataset for each kind of fruit from the training dataset, as specified in the paper.
- binary classification of freshness of a given fruit

## [Fruits and Vegetables dataset](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)

A brand-new dataset without monotone background. Similarly, we propose 2 different dataset configurations.

- Binary classification of freshness of fruit or vegetables. (_**single_rotten.py**_)
- 5-class classification among all kinds of fruits or vegetables.
- 10-class classification among all kinds of fruits and vegetables.

