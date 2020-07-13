# Face Find
Realtime facial recognition using a Siamese Convolutional Neural Network!

# How it Works
Learn about how a Siamese CNN works and how it is implemented here in real time using OpenCV, here: MEDIUM ARTICLE CURRENTLY BEING WRITTEN

# Default Usage
- To use default settings/parameters and CNN, simply run 'train_model.py' with to generate the trained model, and then 'model_evaluate.py' if you want to see the results on the test set.
- To run in real time using your webcam, run 'app/main.py'
- To add known/recognizable people, follow the given example to create folders in 'app/people' (photos should be from the front, and background should be relatively empty - solid color works best). It is best to have 2-3 images of every person.

# Custom Usage
- Pre-created models are in 'models.py' - you can use the same format in there, to add your own custom models
- To use a different model, change 'model = models.get_model_1(input_shape)' in 'train_model.py', using the name of the function from 'models.py'
-  To add known/recognizable people, follow the given example to create folders in 'app/people' (photos should be from the front, and background should be relatively empty - solid color works best). It is best to have 2-3 images of every person.

# Running Files
Running 'train.py' (all params are optional, as there are defaults)
```shell
python3 train.py --epochs NUM_EPOCHS --learning_rate LEARNING_RATE --model_save_name NAME_TO_SAVE.H5
```
Running 'model_evaluate.py' (all params are optional, as there are defaults)
```shell
python3 model_evaluate.py --model_name MODEL_NAME.H5
```
Running 'app/main.py' (all params are optional, as there are defaults)
```shell
python3 app/main.py --model_name MODEL_NAME.H5
```

# Miscellaneous Notes
- Look at 'best_results.MD' for best results
- Currently only supports images with 1 person
- Glasses or other accessories may throw off the model
- Model tends to underperform if person is not directly facing camera
- Currently only supports images where person is direclty facing the camera - a frontal view of their face - because faces are being detected and extracted using OpenCV frontal-face HaarCascade
- There is currently no command line arguement parser implemented, so the changing of values or parameters such as model optimizer must be done inside the '.py' files
- Currently no requirements.txt file
- As noted in import comments, the code uses a Keras backend called PlaidML, since I use an AMD GPU and cannot run Tensorflow. Unless you use PlaidML as well (you will know for sure if you do), get rid of the PlaidMl imports, and import Tensorflow + Keras as you normally would
- Currently working on comprehensive Medium article
- -Currently developing website to implement Siamese CNN for face recognition on your own images
- Both of the above will be added here once available
- Dataset being used: Labelled Faces in the Wild (LFW) - - http://vis-www.cs.umass.edu/lfw/
- Feel free to experiment and create better models!
- If you have any comments or questions, please create an issue or shoot me an email at zarif.azher@gmail.com
