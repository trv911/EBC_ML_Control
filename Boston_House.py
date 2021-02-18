from pandas import read_csv

# load dataset
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
#Y = dataset[:,13]
# print to see if working
print(X)