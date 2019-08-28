# Build Decision Tree Regression, Linear Regression and Polynomial Regression models 

* Using Sklearn for predicting the Violent Crimes Per Population in USA. 

* Dataset: Download this racist dataset “Communities and Crime” from the [link](
https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime) The dataset contains
127 features and target (Violent Crimes Per Population)

### Preprocessing of data 

 - Opening data file and making header
 - Replacing '?' (Missing Values) with np.NaN and filling nan values with mean of the column through Imputer
 - storing Y (Result) column in list and dropping the Y column from our main dataframe
 - doing hot encoding (using pandas dummy)
 - Merging the dummy dataframe with main dataframe
 - Removing the column in which hot encoding was performed (Here Hour Column)
 - Splitting the dataframe in training and testing dataset