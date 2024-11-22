# FET4H: Efficient Hierarchical CNN Model with Self-Attention for Three-Category Facial Emotion Tracking in Healthcare Applications
## Overview
The FET4H system monitors and evaluates patients' emotions for healthcare applications. It can integrate with other systems via APIs or run as a standalone module on client devices. The system captures images or videos, processes them, and recognizes facial emotions, generating timestamped results for storage, evaluation, and visualization over specific periods or interactions.
## Features
The FET4H software offers key functionalities, including face detection, facial emotion recognition, result tracking, and real-time visualization. It captures frames in real-time (e.g., from a webcam or RTSP stream) and uses libraries like MediaPipe to detect faces and extract bounding boxes. The integrated FET4H model classifies emotions (NEGative, NEUtral, POSitive) with confidence scores, displaying results on-screen in real-time and storing them for analysis.
Real-time visualizations include a pie chart showing the emotion distribution and a line chart plotting confidence levels over time, both with dynamic updates and distinct colors. Captured images and detailed logs of detected faces and emotions are saved to a configurable folder for retraining or integration with other systems.
## Installation & usage
To use FET4H (e.g., MACOS), follow these steps:
1. Clone the repository:
<pre> git  clone  https://github.com/duongthanglong/FET4H.git </pre>
2. Install the required dependencies: `pip  install  -r  requirements.txt`
3. Two usage scenarios for the FET4H system (standalone or as a library of the FET4H model in your Python project):
    * Run as standalone: `python   FET4H.py`
    * Import FET4H_model into your Python project, then create an instance of the FET4H model for using facial emotion prediction:
      <pre>   import  tensorflow  as  tf
         from  FET4H_model  import  *
         model = tf.keras.models.load_model('../FET4H_model')
         y_preds = model.predict(#list_of_images#) </pre>    
      where, images in the #list_of_images# are normalized to [-1,1] with the shape of [70,70,3]. The predicted y_preds is an array of probability of emotions coresponding to images, they can be applied tf.argmax to get an emotion for each image.
      
**1. Prerequisites**
### Input rules
### Running applications
### Get outputs

## Examples

## Contributing

## Dataset for training and testing



