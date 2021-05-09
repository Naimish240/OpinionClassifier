# OpinionClassifier

# **Introduction**

In the world of social media, we are being constantly bombarded with textual information left and right. Identifying whether we are reading an _objective statement_ or a _subjective opinion_ becomes difficult. But why is it important, to distinguish between the two?

To answer this, we must first understand the difference between the two.

---
- **Objective Statement** : A detail which may or may not be true, depending on objective observed proof that anyone can verify.

- **Subjective Opinion** : A biased interpretation / judgement that can neither be proved nor disproved.
---

Let us take the following example: **"_Dairy Milk Silk is the best chocolate_"** 

Although it _sounds_ like an objective statement, it is an opinion. And an opinion that can lead to arguments, if another person comes along claiming that a 5 star chocolate is better. It is not that hard to imagine how similar statements (on topics like politics) can lead to chaos both online and offline. Hence, there exists a need to clearly identify a statement as an **opinion** to predict and de esecalate potential conflicts.

--- 

The next natural question would be, how do we identify an opinion?

One method is through asking the following question:
- Does it contain words that seem to express bias?

If the answer for this question is "yes", then it is an opinion.

---
Now that we have understood how we humans can identify an opinion, we can next come to the crux of this project. 

**_"How do we get a computer to figure out if a given statement is an opinion?"_**

People have attempted to solve this problem in the past, as follows. 

- [Akshat Bakliwal et al.](https://www.aclweb.org/anthology/W11-3715.pdf) made use of a "Simple + POS-Tagged NGram with Negation Handling Feature MLP model", and achieved an accuracy of 81.60% on the Movie Reviews dataset. 
- [Mateusz Tatusko](https://github.com/espressoctopus/opinion-or-fact-sentence-classifier) used a Bag of Words (BOW) based approach with the Opiniosis dataset, and got 97% accuracy in training. But the model did not perform well on samples outside the dataset. 
- [Mahmoud Othman et al.](http://www.jcomputers.us/vol11/jcp1105-05.pdf) used Part of Speech (POS) tagging on a multi class dataset of 4000 statements, and achieved an average precision of 82.525% across all 4 classes.
- [Jorge Carrillo-de-Albornoz et al.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0209961) have also worked on extracting facts, opinions and experiences from health forms. They used a Support Vector Machine (SVM) on a dataset of 3000 posts using 10-fold cross validation to achieve an accuracy of above 80%.

---

In the notebook `OpinionClassifier.ipynb`, I examine the potential of various machine learning algorithms for solving this task. The algorithms I have tested include

- Logistic Regression
- Random Forest
- SVC
- Naive Bayes Classifier
- LSTM (Defined using Sub Classing) and
- 1D CNN (Defined using Sequential Method)

To vectorize the inputs, I have made use of TF-IDF for the ML models and Embedding Layers for the DL models.

Subsequently, I have also made use of Shapley Additive explanations (SHAP) to interpret the results of the various models.

Following the results of the notebook, I have also made a [website]() hosting the model. Users can give a statement as input, and the model will return its prediction for the sentence, along with an image explaining why it made its decision, through Shapley analysis.

---
