# Amazon Reviews Sentiment Analysis Documentation - Team 18
**Introduction:**
 
Amazon reviews serve several important purposes for customers and sellers. Specifically, reviews can help customers make more informed decisions by providing authentic and informed feedback on products. Positive reviews can help highlight a product to potential customers, while negative reviews will dissuade buyers and inform sellers of areas to improve. In totality, reviews add transparency to the buying and selling process on Amazon.
 
For sellers, it is important to quickly analyze customer reviews to assess customer sentiment on their products. For a large volume of reviews, it is infeasible for these reviews to be manually read by humans. It is therefore important to have a machine learning solution to quickly determine the sentiment of customers. Sentiments in the real world can be complex, so it is difficult to identify just one model or algorithm to use. Currently, popular tools for sentiment analysis include SVMs and Logistic Regression that work up to a threshold. Through research and experiments, we determined that a hybrid model is best to tackle challenges currently existing in sentiment analysis models. This project aims to develop an optimized tool for sentiment analysis based on the text of customer reviews.

Video: https://mediaspace.illinois.edu/media/t/1_j7irncc4

**Problem Statement:**
 
In this project, we aim to build and evaluate a novel Sentiment Analysis Tool using a dataset of Amazon reviews. The model will be trained to classify the sentiment of each review as positive or negative using natural language processing, tokenization and/or k-fold cross validation/other methods.
 
For evaluation, the model’s sentiment predictions will be compared with the actual sentiment ratings in the dataset to assess precision and recall. We can also evaluate our work against Amazon reviews outside the given dataset and compare if our sentiment analysis works across sets. Our analysis will focus on commonly occurring words or phrases in reviews with common sentiment. This would be useful because it could be that many positive reviews are associated with certain phrases and vice versa for negative phrases. First, we conducted research into the current state of sentiment analysis as well as the various tools and algorithms being used in practice today. We will compare and contrast the different training methods against the same datasets to determine the most appropriate training methodology. These training methods could include logistic regression, support vector machines, naïve bayes classification, stochastic gradient descent classification, k-nearest neighbors, etc. which we will determine while working throughout the project. 
 
 
**Data and Tools:**

• Google Colab: http://colab.research.google.com/

• Amazon Review data from Kaggle:
https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data for
Sentiment Analysis
 
Selected Python Libraries:
·  	NLTK (Natural Language Toolkit) – This library contains a list of ‘stopwords’ which we remove as part of our data preprocessing. It also contains VADER, which is an existing model for sentiment analysis for single text.
·  	SKLearn – This library contains all the machine learning models which we implemented in our project (SVM, Random Forest, etc.). It also provides the confusion matrix and classification metrics on which we judge the performance of the sentiment analysis. Additionally, the library provided the stacking classifier that allowed us to create a final hybrid model. 
·  	Transformers – This library is used for BERT and RoBERTa embeddings. 
 
**Software / Access**

For this project we used Google Colab. Google Colab is a hosted Jupyter Notebook platform that allows users with Google accounts to share and modify code. To use it, first login using a Google account. Then download the project notebook from Github.
 
https://github.com/satankita/AmazonReviewsSentimentAnalysis
 
Google Colab Link:  https://colab.research.google.com/drive/1ddMoyU_0E7H-v_k5VltzaqVq1Ckr0aPc?usp=sharing
 
To upload our notebook onto your own Colab, do the following:
 
From http://colab.research.google.com/,
Select: File ->Upload Notebook->Upload->Choose File (should be saved in downloads).
  
Download the train.csv and test.csv datasets from Kaggle and upload data sets to your personal Google Drive. If you’d like, save both sets to a folder with a given project name (we used ‘TIS Project/Data’). Copy the file path from the Google Drive and replace it to DATA+DIR = mypath + “ ”, where the final text in quotes is your selected data file folder in Google Collab. For example, our Colab setup looks like:
 
 
When run, Colab will prompt you to connect to your drive and sign in again through your Google account. Simply affirm to continue running the notebook.

Note - in our Google Collab file we have included commented out experiment code that does not contribute to the final model, but is there for reference (the  results of these experiments helped us build the final model). 
 
**Process Flow (how we appraoched our project)**
<img width="799" alt="Screenshot 2024-12-11 at 6 14 02 PM" src="https://github.com/user-attachments/assets/708feaa4-8c04-47fe-b82e-e4cb2689def8" />

**Data Processing**
 
The original data set was used for the Stanford Network Analysis Project (SNAP) and contains 34,686,770 Amazon reviews from 6,643,669 users on 2,441,053 products, spanning 18 years up to March 2013. The subset of data listed on Kaggle contains 1,800,000 training samples and 200,000 testing samples. Reviews include product and user information, ratings, and a plaintext review. These reviews have sentiment labels with class 1 as the negative and class 2 as the positive.
 
The full dataset of text samples and sentiment labels summed up to over 1.5gb of text data, making it too large to process efficiently within the scope of the Jupyter notebook. To address this issue, we decided to randomly sample 1% of the data. This smaller sample allowed us to train our model sufficiently fast while retaining enough data to achieve meaningful insights from our testing.  
 
From above, we used an overall sample of approximately 36,000 Amazon reviews and it is almost evenly split between positive and negative sentiments. There is also a wide range of variation in the words and the length of the reviews in the data. This smaller subset allowed us to train and test our model within the given timeframe for the project.
 
For initial training and testing, we used a 70% training sample and 30% testing sample. The distribution of positive/negative sentiments closely matches that of the overall in both cases.
 
 
**Count Vectorization, TF-IDF Transformation, and VADER Sentiments:**
 
Our initial text processing method was a combination of count vectorization, TF-IDF transformation, and adding VADER sentiments. The count vectorizer uses a custom cleaning function to remove stopwords and separate the review text into individual words as follows:
 
Next, we applied VADER to these individual words. VADER is an existing sentiment analysis model that outputs probabilities of 4 sentiments for each word: Positive, Negative, Neutral and Compound. Positive, Negative, and Neutral range from 0 to 1 scores while Compound takes the aggregate of the previous 3 scores, with 1 being the strong negative sentiment and 1 being the strong positive sentiment. These provide 4 additional predictors to the dataset.
 
Last, we applied TF-IDF transform to account for issues of term frequency and document frequency between reviews. This gave us an initial predictor list of over 89000, encoded as TF-IDF weights and 4 VADER scores.

We found that Tf-IDF did not contribute to optimal results, so we did not move further with it.
 
 
**Vector Embeddings:**

To enhance the representation of text data, we used BERT models. Text embeddings were generated using Both BERT-base and RoBERTa. BERT (Bidirectional Encoder Representations from Transformers) is a pretrained model that captures contextual word semantics  in high-dimensional vector space. Text data was tokenized using BertTokenizer, and for each review, the [CLS] token’s representation was extracted as the sentence embedding. These embeddings further improved the results obtained from the earlier transformations, by about 4 percentage points (this discovery was found after initial tests of different algorithms; applying vector embeddings dramatically improved the results). 

Additionally, we used RoBERTa (Robustly Optimized BERT Approach), an optimized version of BERT. Like BERT, we tokenized the text with RobertaTokenizer and processed with RobertaModel; the [CLS] tokens were extracted from each review to act as our features. RoBERTa embeddings were shown to outperform BERT embeddings by approximately 5 percentage points, demonstrating their ability to capture greater contextual representations. To augment these embeddings, we used features like text length and word count.

In our testing, we found that RoBERTa embeddings out performed BERT embeddings, which makes sense, as RoBERTa is an optimized version of BERT. 
 
**Machine Learning Models:**

Results for each classification are measured in the notebook with the classification report function from SkLearn which provides precision, recall, F1-score, and accuracy.

Support Vector Machine
 
Support vector machine (SVM) is a supervised machine learning algorithm for classification. It aims to find the optimal hyperplane that best separates data into different classes in a high-dimensional space. In our project we used the LinearSVC function in SkLearn, which uses a linear kernel and is specific for classification. We used the default parameters, which uses L2 penalty and squared-hinge loss. Some key advantages of SVM is that it is good with high-dimensions - which our data has after embedding, its inherent property to be less prone to overfitting and its versatility to work with different pre-processing pipelines (including TF-IDF, bag-of-words, etc).

XGBoost
 
XGBoost (Extreme Gradient Boosting) is a supervised machine learning algorithm that extends gradient boosting methods for classification. Gradient boosting builds ensembles of weak learners (usually decision trees) in a sequential manner and each new tree attempts to correct the errors of the previous. Gradient refers to the descent, the method of updating model parameters to minimize loss. In addition, XGBoost includes regularization terms to help prevent overfitting when training the model and because it splits the data into decision trees, missing or noisy data can be handled better without relying heavily on pre-processing. In comparison to standard gradient boosting, XGBoost has some key advantages, namely improved scalability, faster training through parallelization, handling of imbalanced data, and built in features for customization of regularization and tree pruning. In our project we used the XGBClassifier function in SkLearn with default parameters and objective = ‘binary:logistic’ which supports 2 class classification.


 
Random Forest
 
Random Forest is a supervised machine learning algorithm for classification that uses multiple decision trees to improve the robustness and accuracy of predictions as opposed to singular decision trees. Random forests are generated using bootstrap aggregation (bagging), the subsampling and training of several decision trees whose predictions are aggregated in the end to produce a final result. Predictors at each level for each subsample are randomly selected to avoid repeated use of the same predictors. Random forest has advantages over decision trees in that it reduces overfitting by averaging the predictions of many decision trees, handles high dimensional data by using subsets of features of each tree, and is generally more accurate. In our project we use the RandomForestClassifier function in SkLearn with default parameters, which sets no limit to the tree depth and uses Gini impurity to measure classification improvement at each split.

K-Nearest Neighbors
 
K-Nearest Neighbors (KNN) is a supervised machine learning algorithm for classification that classifies a data point based on the similarity to the majority of the k nearest points. It requires a distance measure and a selection for how many neighbors to make a classification. The advantages of KNN are that it is simple to understand and use, and that it is nonparametric (requires no assumptions of underlying variables). However, it is sensitive to irrelevant features, and is poor in high dimensions. In high dimensional feature spaces, we would need to introduce feature reduction techniques (PCA, etc) to improve the accuracy/effectiveness of the model at the cost of adding pre-processing complexity. In addition, figuring out the correct choice of k can heavily affect the model’s performance, as smaller k values cause the model to be more sensitive to noise and larger k values may dilute the importance of certain features/data. In our project we used the KNeighborsClassification in the SkLearn library with default parameters, which sets k=5, and the distance measure to be Euclidean distance.


After running these tests, we analyzed the results to determine which models performed best. 

**Final Product:**
Based on the results of our research and experiments, we found that the optimal embedding model was roBERTa, which out performed BERT by about 4%. In earlier tests, we found that embeddings in general improved the performance of the sentiment analysis in tests comparing algorithms with and without embeddings applied.

RoBERTa combined with SVM yielded the highest results. Combining this with the additional features from VADER optimized the accuracy of the model. 

XGBoost also has extremely high results, almost equating to the results of RoBERTa combined with SVM. For this reason, we suggest a stacked ensemble model of RoBERTa, SVM, XGBoost, and Logistic Regression with VADER features for optimal results. SVM works well with high-dimensional data with linear relationships. XGBoost is able to then fill gaps with optimizing for nonlinear relationships in the data (improving robustness). A logistic regression model is included in the stacking ensemble as it is able to combine predictions from base models SVM and XGBoost effectively. So although our stacked model results are similar to the SVM results, this hybrid model is optimal for adapting future types of linear and nonlinear relationships and varying data sets. 


<img width="777" alt="Screenshot 2024-12-11 at 6 01 31 PM" src="https://github.com/user-attachments/assets/54a4d9d3-dbd2-4ac2-85c0-f3e4d2f0a5c1" />




