# Dog_Breed_Detection_Web_App_Using_Deep_Learning

This web app includes HTML, CSS and JS for frontend and Keras, Flask with dependencies for backend. The web app detect a dog breed using deep learning with automated image pre-processing and internal procedures. 

Pretrained deep learning for dog breed detection is stored as model_60_epochs_adam_02.h5 and model_60_epochs_adam_02.json.I used transfer learning for the deep learning model training.The accuracy of the model is approximately 98%.

For deploying I decided to use Flask instead of Django because of it simplicity and compatibility with Amazon Web Services (AWS) and Google Cloud Storage (GCS). 
For using this web app in your personal desktop simply clone the repository and install dependencies in requirements.txt file using  <h5>pip install -r requirements.txt</h5> and run the application using 
<h5>python app.py</h5> in shell.
