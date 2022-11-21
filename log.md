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

