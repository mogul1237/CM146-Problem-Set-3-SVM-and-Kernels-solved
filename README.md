# CM146-Problem-Set-3-SVM-and-Kernels-solved

Download Here: [CM146 Problem Set 3: SVM and Kernels solved](https://jarviscodinghub.com/assignment/problem-set-3-svm-and-kernels-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

1 Kernels [8 pts]
(a) For any two documents x and z, define k(x, z) to equal the number of unique words that occur
in both x and z (i.e., the size of the intersection of the sets of words in the two documents).
Is this function a kernel? Give justification for your answer.
(b) One way to construct kernels is to build them from simpler ones. We have seen various
“construction rules”, including the following: Assuming k1(x, z) and k2(x, z) are kernels,
then so are
• (scaling) f(x)k1(x, z)f(z) for any function f(x) ∈ R
• (sum) k(x, z) = k1(x, z) + k2(x, z)
• (product) k(x, z) = k1(x, z)k2(x, z)
Using the above rules and the fact that k(x, z) = x · z is (clearly) a kernel, show that the
following is also a kernel:

1 + 
x
||x||
·

z
||z||3
(c) Given vectors x and z in R
2
, define the kernel kβ(x, z) = (1 + βx · z)
3
for any value β > 0.
Find the corresponding feature map φβ(·)
1
. What are the similarities/differences from the
kernel k(x, z) = (1 + x · z)
3
, and what role does the parameter β play?
2 SVM [8 pts]
Suppose we are looking for a maximum-margin linear classifier through the origin, i.e. b = 0 (also
hard margin, i.e., no slack variables). In other words, we minimize 1
2
||θ||2
subject to ynθ
Txn ≥
1, n = 1, . . . , N.
Parts of this assignment are adapted from course material by Tommi Jaakola (MIT), and Andrew Ng (Stanford),
and Jenna Wiens (UMich).
1You may use any external program to expand the cubic.
1
(a) Given a single training vector x = (a, e)
T with label y = −1, what is the θ
∗
that satisfies the
above constrained minimization?
(b) Suppose we have two training examples, x1 = (1, 1)T and x2 = (1, 0)T with labels y1 = 1 and
y2 = −1. What is θ
∗
in this case, and what is the margin γ?
(c) Suppose we now allow the offset parameter b to be non-zero. How would the classifier and the
margin change in the previous question? What are (θ
∗
, b∗
) and γ? Compare your solutions
with and without offset.
3 Twitter analysis using SVMs [26 pts]
In this project, you will be working with Twitter data. Specifically, we have supplied you with a
number of tweets that are reviews/reactions to movies2
,
e.g., “@nickjfrost just saw The Boat That Rocked/Pirate Radio and I thought it was brilliant! You
and the rest of the cast were fantastic! < 3”.
You will learn to automatically classify such tweets as either positive or negative reviews. To do
this, you will employ Support Vector Machines (SVMs), a popular choice for a large number of
classification problems.
Download the code and data sets from the course website. It contains the following data files:
• tweets.txt contains 630 tweets about movies. Each line in the file contains exactly one
tweet, so there are 630 lines in total.
• labels.txt contains the corresponding labels. If a tweet praises or recommends a movie, it
is classified as a positive review and labeled +1; otherwise it is classified as a negative review
and labeled −1. These labels are ordered, i.e. the label for the i
th tweet in tweets.txt
corresponds to the i
th number in labels.txt.
• held_out_tweets.txt contains 70 tweets for which we have withheld the labels.
Skim through the tweets to get a sense of the data.
The python file twitter.py contains skeleton code for the project. Skim through the code to
understand its structure.
3.1 Feature Extraction [2 pts]
We will use a bag-of-words model to convert each tweet into a feature vector. A bag-of-words
model treats a text file as a collection of words, disregarding word order. The first step in building
a bag-of-words model involves building a “dictionary”. A dictionary contains all of the unique
words in the text file. For this project, we will be including punctuations in the dictionary too.
For example, a text file containing “John likes movies. Mary likes movies2!!” will have a dictionary {’John’:0, ’Mary’:1, ’likes’:2, ’movies’:3, ’movies2’:4, ’.’:5, ’!’:6}. Note
that the (key,value) pairs are (word, index), where the index keeps track of the number of
unique words (size of the dictionary).
2Please note that these data were selected at random and thus the content of these tweets do not reflect the views
of the course staff. 🙂
2
Given a dictionary containing d unique words, we can transform the n variable-length tweets into
n feature vectors of length d by setting the i
th element of the j
th feature vector to 1 if the i
th
dictionary word is in the j
th tweet, and 0 otherwise.
(a) We have implemented extract_words(…) that processes an input string to return a list of
unique words. This method takes a simplistic approach to the problem, treating any string
of characters (that does not include a space) as a “word” and also extracting and including
all unique punctuations.
Implement extract_dictionary(…) that uses extract_words(…) to read all unique
words contained in a file into a dictionary (as in the example above). Process the tweets in
the order they appear in the file to create this dictionary of d unique words/punctuations.
(b) Next, implement extract_feature_vectors(…) that produces the bag-of-words representation of a file based on the extracted dictionary. That is, for each tweet i, construct a
feature vector of length d, where the j
th entry in the feature vector is 1 if the j
th word in the
dictionary is present in tweet i, or 0 otherwise. For n tweets, save the feature vectors in a
feature matrix, where the rows correspond to tweets (examples) and the columns correspond
to words (features). Maintain the order of the tweets as they appear in the file.
(c) In main(…), we have provided code to read the tweets and labels into a (630, d) feature
matrix and (630,) label array. Split the feature matrix and corresponding labels into your
training and test sets. The first 560 tweets will be used for training and the last 70
tweets will be used for testing. **All subsequent operations will be performed on these
data.**
3.2 Hyperparameter Selection for a Linear-Kernel SVM [10 pts]
Next, we will learn a classifier to separate the training data into positive and negative tweets. For
the classifier, we will use SVMs with two different kernels: linear and radial basis function (RBF).
We will use the sklearn.svm.SVC class and explicitly set only three of the initialization parameters:
kernel, gamma, and C. As usual, we will use SVC.fit(X,y) to train our SVM, but in lieu of using
SVC.predict(X) to make predictions, we will use SVC.decision_function(X), which returns the
(signed) distance of the samples to the separating hyperplane.
SVMs have hyperparameters that must be set by the user. For both linear and RBF-kernel SVMs,
we will select the hyperparameters using 5-fold cross-validation (CV). Using 5-fold CV, we will
select the hyperparameters that lead to the ‘best’ mean performance across all 5 folds.
(a) The result of a hyperparameter selection often depends upon the choice of performance measure. Here, we will consider the following performance measures: accuracy, F1-Score,
AUROC, precision, sensitivity, and specificity.
Implement performance(…). All measures, except sensitivity and specificity, are implemented in sklearn.metrics library. You can use sklearn.metrics.confusion_matrix(…)
to calculate the other two.
(b) Next, implement cv_performance(…) to return the mean k-fold CV performance for the
performance metric passed into the function. Here, you will make use of SVC.fit(X,y) and
SVC.decision_function(X), as well as your performance(…) function.
3
You may have noticed that the proportion of the two classes (positive and negative) are not
equal in the training data. When dividing the data into folds for CV, you should try to keep
the class proportions roughly the same across folds. In your write-up, briefly describe why
it might be beneficial to maintain class proportions across folds. Then, in main(…), use
sklearn.cross_validation.StratifiedKFold(…) to split the data for 5-fold CV, making
sure to stratify using only the training labels.
(c) Now, implement select_param_linear(…) to choose a setting for C for a linear SVM based
on the training data and the specified metric. Your function should call cv_performance(…),
passing in instances of SVC(kernel=’linear’, C=c) with different values for C, e.g., C =
10−3
, 10−2
, . . . , 102
.
(d) Finally, using the training data from Section 3.1 and the functions implemented here, find
the best setting for C for each performance measure mentioned above. Report your findings
in tabular format (up to the fourth decimal place):
C accuracy F1-score AUROC precision sensitivity specificity
10−3
10−2
10−1
100
101
102
best C
Your select_param_linear(…) function returns the ‘best’ C given a range of values. How
does the 5-fold CV performance vary with C and the performance metric?
3.3 Hyperparameter Selection for an RBF-kernel SVM [8 pts]
Similar to the hyperparameter selection for a linear-kernel SVM, you will perform hyperparameter
selection for an RBF-kernel SVM.
(a) Describe the role of the additional hyperparameter γ for an RBF-kernel SVM. How does γ
affect generalization error?
(b) Implement select_param_rbf(…) to choose a setting for C and γ via a grid search. Your
function should call cv_performance(…), passing in instances of
SVC(kernel=’rbf’, C=c, gamma=gamma) with different values for C and gamma. Explain
what kind of grid you used and why.
(c) Finally, using the training data from Section 3.1 and the function implemented here, find
the best setting for C and γ for each performance measure mentioned above. Report your
findings in tabular format. This time, because we have a two-dimensional grid search, report
only the best score for each metric, along with the accompanying C and γ setting.
4
metric score C γ
accuracy
F1-score
AUROC
precision
sensitivity
specificity
How does the CV performance vary with the hyperparameters of the RBF-kernel SVM?
3.4 Test Set Performance [6 pts]
In this section, you will apply the two classifiers learned in the previous sections to the test data
from Section 3.1. Once you have predicted labels for the test data, you will measure performance.
(a) Based on the results you obtained in Section 3.2 and Section 3.3, choose a hyperparameter
setting for the linear-kernel SVM and a hyperparameter setting for the RBF-kernel SVM.
Explain your choice.
Then, in main(…), using the training data extracted in Section 3.1 and SVC.fit(…),
train a linear- and an RBF-kernel SVM with your chosen settings.
(b) Implement performance_test(…) which returns the value of a performance measure, given
the test data and a trained classifier.
(c) For each performance metric, use performance_test(…) and the two trained linear- and
RBF-kernel SVM classifiers to measure performance on the test data. Report the results. Be
sure to include the name of the performance metric employed, and the performance on the
test data. How do the test performance of your two classifiers compare?
