# Best Results
- The best I've been able to obtain in terms of accuracy is about 65% on the test set, using model 1 with 30 epochs and .00005 learning rate. However, it was definitely very overfit, and model 2 (.0001 learning rate) which performs better in that regard, has 62% on the test set and a much lower validation loss. Please let me know (see README for email) if you achieve better!
- Model 2 seems to work best in real time
- Test set accuracy of around 60% is fine, because in real time the model must determine that all images of a person match the one in the webcam frame, before declaring that it recognizes a certain individual
- Threshold of 50-55 seems to work best
- Overall best results for me has been with model 2, .0001 learning rate with 37 epochs.
