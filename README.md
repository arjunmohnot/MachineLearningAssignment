
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
## What is Machine Learning?


> > - “Machine learning is the next Internet” -Tony Tether, Director, DARPA
---
> > - “Machine learning is the hot new thing” -John Hennessy, President, Stanford
---
> > - “Machine learning is today’s discontinuity” -Jerry Yang, CEO, Yahoo
---
> > - “Machine learning is the new electricity”-Andrew Ng, Chief Scientist Baidu
---

| Table of Content                                                  |
|-------------------------------------------------------------------|
| 1. Assignment 1 - Getting familiar with Data types and Visualization |
| 2. Assignment 2 - Decision Tree Part 1, 2 - ID3, Cart                |
|                                                                   |   

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
- Download 1 classification dataset (https://tinyurl.com/uciclass) and 1 regression
dataset (https://tinyurl.com/ucireg) of your choice. Each of you should have unique
datasets with you.
- Test the CART model with the samples and calculate the accuracy. Print the
decision path for each of the samples.
- Load the data, pre-process the data. Split the dataset into training and testing sets
using built-in sklearn functions. Build a CART model for classification as well as
regression and do the procedure as in part 1.
