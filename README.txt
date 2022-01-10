Team members:
Nathan Johnson (njohns60@calpoly.edu)
Edward Zhou (ekzhou@calpoly.edu)

File Names:
InduceC45.py - C4.5 file
classifier.py - uses C4.5 to create a classifier
validation.py - cross validation script
randomForest.py - random forest implementation
knn.py - k-nearest neighbors implementation

Code files can be found in /src
Output files can be found in /output with the name <dataset name>-<model name>-results.out

Hyperparameters:
	Iris
		Decision tree: threshold - 0.2
		RF: attributes - 2, trees - 50, observations - 85
		KNN: k - 12
	Letter
		Decision tree: threshold - 0.1
		RF: attributes - 10, trees - 50, observations - 400
		KNN: k - 7
	Red Wine
		Decision tree: threshold - 0.2
		RF: attributes - 4, trees - 30, observations - 100
		KNN: k - 2
	White Wine
		Decision tree: threshold - 0.2
		RF: attributes - 5, trees - 30, observations - 125
		KNN: k - 2
	Credit Approval
		Decision tree: threshold - 0.2
		RF: attributes - 4, trees - 40, observations - 100
		KNN: k - 2
	Heart
		Decision tree: threshold - 0.1
		RF: attributes - 5, trees - 50, observations - 125
		KNN: k - 2

Restirction File Structure: columns that should be used are specified by 1, columns that are not are specified by 0. Seperate each value by new line.
Example:
1
0
1
1

Json Structure:
According to the lab specification, though we used ordered dictionaries.

How to Run:
It is advised to use validation.py to run the script, which takes in the parameters:
<data file> <folds> <model name> <restriction file> <model parameters (# neighbors for KNN or # trees for RF)>