
# Machine Learning Assignments (ECSE-303L)

<img src="https://avatars0.githubusercontent.com/u/33459977?s=80&v=4" align="left"/>

---------------------------------------------------------------------------
About
=====

- **ARJUN MOHNOT**
- Mobile: +91-7733993964
- [Website](https://arjun009.github.io)
- [WhatsApp](https://wa.me/917733993964?text=Hey%20Arjun%20Mohnot,%20I%27m%20contacting%20you%20from%20your%20Github%20Repository,%20A.I.-M.L.)
- [PlayStore](https://play.google.com/store/apps/developer?id=ARJUN+MOHNOT)

---------------------------------------------------------------------------

<img width="226" alt="Capture" src="https://user-images.githubusercontent.com/33459977/62526941-d9f6b380-b857-11e9-8a93-048baa5b4243.PNG">


## What is Machine Learning?

- ` Machine Learning is the study of algorithms that •improve their performance P•at some task T•with experience E.A well-defined learning task is given by <P, T, E>.`

> - “Machine learning is the next Internet” -Tony Tether, Director, DARPA
---
> - “Machine learning is the hot new thing” -John Hennessy, President, Stanford
---
> - “Machine learning is today’s discontinuity” -Jerry Yang, CEO, Yahoo
---
> - “Machine learning is the new electricity”-Andrew Ng, Chief Scientist Baidu

> ![65950590_702544676861606_2656840745836301205_n](https://user-images.githubusercontent.com/33459977/62526444-e29aba00-b856-11e9-856b-a817c0461b05.jpg)

---

| Table of Content                                                  |
|-------------------------------------------------------------------|
| 1. Assignment 1 - Getting familiar with Data types and Visualization |
| 2. Assignment 2 - Decision Tree Part 1, 2 - ID3, Cart                |
| 3. Assignment 3 - Linear Regression (Gradient Descent), Polynomial Regression and Decision Tree Regression       |   

Assignment 1
================

- Table data visualization: line graph, bar graph, histogram chart, pie chart, scatter plot.
- Image visualization: image plot, 3d plot.
- Video visualization: video player. 
- Audio visualization: audio player, spectrogram.
- Text visualization: Word cloud, bubble cloud

Assignment 2
=================

ID3
====

- Compute the Information Gains (using difference in weighted entropy) of the first
features that better discriminates the single vs committed. Then split the tree based
on the first feature.
- Choose the further features based on the information gains. Don’t the split the tree if
the information gain is not optimal.
- Print the information gains for each feature and every split. Finally print the tree.
- Test the model with the below samples and calculate the accuracy.

Cart
=====

- Use the Sklearn package to implement the CART Decision tree for the following data.
After training, finally visualize the tree, print the importance of features (Gini values),
properties of the tree such as number of leaves, depth of the tree etc.
- Download 1 classification dataset [uciclass](https://tinyurl.com/uciclass) and 1 regression
dataset [ucireg](https://tinyurl.com/ucireg) of your choice. Each of you should have unique
datasets with you.
- Test the CART model with the samples and calculate the accuracy. Print the
decision path for each of the samples.
- Load the data, pre-process the data. Split the dataset into training and testing sets
using built-in sklearn functions. Build a CART model for classification as well as
regression and do the procedure as in part 1.

Assignment 3
================

- Encoding
- Normalization
- Data Splitting
- Linear Regression Training: model based on gradient descent algorithm. Automatically initialize a linear regression model with n+1 parameters (n features and 1 bias)
- Linear Regression, Polynomial Regression, Decision Tree Regression (Cart) using sklearn
- Testing: Test the model with the test data and compute the mean squared error (MSE) for
test data. 
- Playing with the Model: You can try different strategies to see whether testing error comes
down or not. Strategies can be different 
  1. Initialization of parameters 
  2. Encoding of features,
  3. removal of some features, 
  4. normalization methods, 
  5. Shuffling of training samples. Check the model error for the testing data for each setup.
