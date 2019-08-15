**Article Courtesy** : [Jason Brownlee](https://machinelearningmastery.com/linear-regression-tutorial-using-gradient-descent-for-machine-learning/)

Linear Regression
===

Stochastic Gradient Descent is an important and widely used algorithm in machine learning.

# Tutorial Data Set
	The data set is completely made up.
	Here is the raw data. The attribute x is the input variable and y is the output variable that is be to predicted by the model.


| X | Y |
|---|---|
| 1 | 1 |
| 2 | 3 |
| 4 | 3 |
| 3 | 2 |
| 5 | 5 |
	

- Below is a simple scatter plot of x versus y.
	![lr](https://user-images.githubusercontent.com/33459977/63126611-dee10300-bfcd-11e9-88f8-bc24bc0642c9.png)

The relationship between x and y looks kind-of linear. As in, this could probably draw a line somewhere diagonally from the bottom left of the plot to the top right to generally describe the relationship between the data. This is a good indication that using linear regression might be appropriate for this little dataset.

# Simple Linear Regression
- When you have a single in put attribute (x) and you want to use linear regression, this is called simple linear regression.
- With simple linear regression it will create the model as follows:
	y = B0 + B1 * x
- This is a line where y is the output variable that is to be predicted, x is the input variable that is know and B0 and B1 are coefficients that are needed to be estimated.
- B0 is called the intercept because it determines where the line intercepts the y axis. In machine learning this can be call the bias, because it is added to offset all predictions that we make. The B1 term is called the slope because it defines the slope of the line or how x translates into a y value before adding our bias.
- The model is called Simple Linear Regression because there is only one input variable (x). If there were more input variables (e.g. x1, x2, etc.) then this would be called multiple regression.

# Stochastic Gradient Descent
- Gradient Descent is the process of minimizing a function by following the gradients of the cost function.
- This involves knowing the form of the cost as well as the derivative so that from a given point you know the gradient and can move in that direction, e.g. downhill towards the minimum value.
- In Machine learning similar technique called stochastic gradient descent to minimize the error of a model on our training data.
- The way this works is that each training instance is shown to the model one at a time. The model makes a prediction for a training instance, the error is calculated and the model is updated in order to reduce the error for the next prediction.
- This procedure can be used to find the set of coefficients in a model that result in the smallest error for the model on the training data. Each iteration the coefficients, called weights (w) in machine learning language are updated using the equation:
	w = w – alpha * delta
- Where w is the coefficient or weight being optimized, alpha is a learning rate that you must configure (e.g. 0.1) and gradient is the error for the model on the training data attributed to the weight.

# Simple Linear Regression with Stochastic Gradient Descent
- The coefficients used in simple linear regression can be found using stochastic gradient descent.
- Linear regression is a linear system and the coefficients can be calculated analytically using linear algebra. Stochastic gradient descent is not used to calculate the coefficients for linear regression in practice (in most cases).
- Linear regression does provide a useful exercise for learning stochastic gradient descent which is an important algorithm used for minimizing cost functions by machine learning algorithms.
- As stated above, linear regression model is defined as follows:
	y = B0 + B1 * x

# Gradient Descent Iteration #1
	- Let’s start with values of 0.0 for both coefficients.
	- B0 = 0.0
	- B1 = 0.0
	- y = 0.0 + 0.0 * x
	- Calculate the error for a prediction as follows:
	- error = p(i) – y(i)
	- Where p(i) is the prediction for the i’th instance in our dataset and y(i) is the i’th output variable for the instance in the dataset.
	- Calculate the predicted value for y using the starting point coefficients for the first training instance:
	- x=1, y=1
	- p(i) = 0.0 + 0.0 * 1
	- p(i) = 0
	- Using the predicted output, calculate the error:
	- error = 0 – 1
	- error = -1
	- Now use this error in our equation for gradient descent to update the weights. This will start with updating the intercept first, because it is easier.
	- B0 is accountable for all of the error. This is to say that updating the weight will use just the error as the gradient. Calculate the update for the B0 coefficient as follows:
	- B0(t+1) = B0(t) – alpha * error
	- Where B0(t+1) is the updated version of the coefficient will use on the next training instance, B0(t) is the current value for B0 alpha is our learning rate and error is the error. Calculate for the training instance. Let’s use a small learning rate of 0.01 and plug the values into the equation to work out what the new and slightly optimized value of B0 will be:
	B0(t+1) = 0.0 – 0.01 * -1.0
	B0(t+1) = 0.01
	- Now, let’s look at updating the value for B1. Use the same equation with one small change. The error is filtered by the input that caused it.Update B1 using the equation:
	B1(t+1) = B1(t) – alpha * error * x
	- Where B1(t+1) is the update coefficient, B1(t) is the current version of the coefficient, alpha is the same learning rate described above, error is the same error calculated above and x is the input value.
	- Plug in our numbers into the equation and calculate the updated value for B1:
	B1(t+1) = 0.0 – 0.01 * -1 * 1
	B1(t+1) = 0.01
	- Finished the first iteration of gradient descent and we have updated our weights to be B0=0.01 and B1=0.01. This process must be repeated for the remaining 4 instances from our dataset.
	- One pass through the training dataset is called an epoch.

# Gradient Descent Iteration #20
	- Let’s jump ahead.
	- You can repeat this process another 19 times. This is 4 complete epochs of the training data being exposed to the model and updating the coefficients.
	- Here is a list of all of the values for the coefficients over the 20 iterations that you should see:

| B0 | B1 |
|----|----|
| 0.01 |	0.01|
| 0.0397 |	0.0694|
| 0.066527 |	0.176708|
| 0.08056049 |	0.21880847|
| 0.1188144616 |	0.410078328|
| 0.1235255337 |	0.4147894001|
| 0.1439944904 |	0.4557273134|
| 0.1543254529 |	0.4970511637|
| 0.1578706635 |	0.5076867953|
| 0.1809076171 |	0.6228715633|
| 0.1828698253 |	0.6248337715|
| 0.1985444516 |	0.6561830242|
| 0.2003116861 |	0.6632519622|
| 0.1984110104 |	0.657549935|
| 0.2135494035 |	0.7332419008|
| 0.2140814905 |0.7337739877|
| 0.2272651958 |	0.7601413984|
| 0.2245868879 |	0.7494281668|
| 0.219858174 |	0.7352420252|
| 0.230897491 |	0.7904386102|

20 iterations or 4 epochs is a nice round number and a good place to stop. You could keep going if you wanted.

Your values should match closely, but may have minor differences due to different spreadsheet programs and different precisions. You can plug each pair of coefficients back into the simple linear regression equation. This is useful because it can calculate a prediction for each training instance and in turn calculate the error.

Below is a plot of the error for each set of coefficients as the learning process unfolded. This is a useful graph as it shows us that error was decreasing with each iteration and starting to bounce around a bit towards the end.

![lr](https://user-images.githubusercontent.com/33459977/63127359-c83bab80-bfcf-11e9-918e-c0066e02bc18.png)

You can see that final coefficients have the values B0=0.230897491 and B1=0.7904386102

Let’s plug them into simple linear Regression model and make a prediction for each point in training dataset.

| X | Y | Prediction |
|---|---|------------|
| 1 | 1 |0.9551001992|
| 2 | 3 |1.690342224 |
| 4 | 3 |3.160826275 |
| 3 | 2 |2.42558425  |
| 5 | 5 |3.8960683   |

Plot dataset again with these predictions overlaid (x vs y and x vs prediction). Drawing a line through the 5 predictions gives us an idea of how well the model fits the training data.
![lr](https://user-images.githubusercontent.com/33459977/63127501-21a3da80-bfd0-11e9-979f-50cb0c39092a.png)
