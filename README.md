# OpinionClassifier

**_"How do we get a computer to figure out if a given statement is an opinion?"_**

Through NLP, we can build a classifier to identify if a given statement is an opinion or not. Dataset for this project was taken from [Alberto Cetoli's](https://github.com/fractalego/subjectivity_classifier) repository. The training and model testing phase has been documented in the form of a Jupyter Notebook, which can be found in the [notebook](https://github.com/Naimish240/OpinionClassifier/tree/main/notebook) folder.

I have built a [website](https://naimish240.github.io/OpinionClassifier/) hosting the model. Users can give a statement as input, and the model will return its prediction for the sentence, along with an image explaining why it made its decision, through Shapley analysis.

---

Tech used:

1. ML/DL Stuff
    - Jupyter Notebooks (For Prototyping the the models)
    - Pandas (For EDA)
    - Sci-kit Learn, TensorFlow/Keras (Building ML and DL models)
    - Shap (To interpret model results)

2. Backend Stuff
    - Flask (For building an API server)
    - Flask-CORS (To Enable CORS, since backend and front end are hosted on different services)
    - Gunicorn (For Web Deployment)
    - Heroku (Hosting the Flask API server)
    - Postman (Testing the API and debugging it)

3. Frontend Stuff
    - HTML, Bootstrap (For website design)
    - JavaScript (For performing Requests, and navigationg DOM to update content)
    - GitHub Pages (For hosting the front end site)
