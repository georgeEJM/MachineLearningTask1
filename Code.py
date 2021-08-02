#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TASK 
# Importing the various modules needed for polynomial regression.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

regData = pd.read_csv("Task1Data.csv") #The file path for the regression data.


# In[3]:


#The data is split into two variables for plotting: X and Y.
Data = regData.iloc[:, 1:3]
#The data is then sorted to accommodate plotting once polynomial regression has been done.
Data.sort_values(by=['x'], inplace=True) 

#Plotting the data initially.
plt.plot(Data['x'], Data['y'], 'bo', label="True Data") #Plots data onto a graph, assigns it a colour, and labels it.
plt.xlabel("X") #Sets the X label
plt.ylabel("Y") #Sets the Y label
plt.title("X and Y data from 'task1.csv'")
plt.legend(loc="lower right") #Uses labels to create a legend.
plt.show() #Shows the graph.


# In[4]:


def pol_regression(features_train, y_train, degree): #Function to compute polynomial coefficients.
    if(degree <= 0):
        degree = 1
    X = np.ones(features_train.shape) #Creates a premade array to increase computational speed.
    for i in range(1,degree+1): #To the nth degree.
       X = np.column_stack((X, features_train ** i)) #Creates an array of X values multiplied by power of i (to n).  
    XX = X.transpose().dot(X) #Creates a Vandermode matrix from X.
    w = np.linalg.solve(XX, X.transpose().dot(y_train)) #Creates polynomial weights.
    pol_coefficients = X.dot(w) #Creates polynomial coefficients.
    return pol_coefficients #Returns them.

y_1 = pol_regression(Data['x'], Data['y'], 4) # 1st degree regression.

def interpolate(x, y, n): # A method to interpolate the data. This can be used if more data is required to test the regression.
    tempX = [] # Creates temporary x and y arrays.
    tempY = []
    for j in range(n): # Interpolate to nth degree
        for i in range(len(x) - 1): # For every value in the data
            d = cm.sqrt((x[i] - x[i+1]**2) + ((y[i] - y[i+1])**2)) # Creates interpolated relationship.

            x1 = x[i] + (i/d) * (x[i+1] - x[i]) # Creates interpolated values.
            y1 = y[i] + (i/d) * (y[i+1] - y[i])

            tempX.append(x1.real) # Adds true and interpolated values to temporary arrays.
            tempX.append(x[i].real)
            
            tempY.append(y1.real)
            tempY.append(y[i].real)
   
    newX = np.array(tempX) # Returns new arrays with interpolated values.
    newY = np.array(tempY)
    return newX, newY


# In[5]:


#Polynomial Regression of the 1st degree.
y_1 = pol_regression(Data['x'], Data['y'], 1) # 1st degree regression.
plt.plot(Data['x'], y_1, 'purple', label='1st degree regression') #Plots regression data.
plt.plot(Data['x'], Data['y'], 'bo', label='True data') #Plots true data.
plt.xlabel("X")
plt.ylabel("Y")
plt.title("1st Degree Polynomial Regression")
plt.legend(loc='lower right')
plt.show()


# In[6]:


#Polynomial Regression of the 2nd degree.
y_2 = pol_regression(Data['x'], Data['y'], 2) #2nd degree regression.
plt.plot(Data['x'], y_2, 'purple', label='2nd degree regression')
plt.plot(Data['x'], Data['y'], 'bo', label='True data')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2nd Degree Polynomial Regression")
plt.legend(loc='lower right')
plt.show()


# In[7]:


#Polynomial Regression of the 3rd degree.
y_3 = pol_regression(Data['x'], Data['y'], 3)#3rd degree regression.
plt.plot(Data['x'], y_3, 'purple', label='2nd degree regression')
plt.plot(Data['x'], Data['y'], 'bo', label='True data')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("3rd Degree Polynomial Regression")
plt.legend(loc='lower right')
plt.show()


# In[8]:


#Polynomial Regression of the 6th degree.
y_6 = pol_regression(Data['x'], Data['y'], 6)#6th degree regression.
plt.plot(Data['x'], y_6, 'purple', label='2nd degree regression')
plt.plot(Data['x'], Data['y'], 'bo', label='True data')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("6th Degree Polynomial Regression")
plt.legend(loc='lower right')
plt.show()


# In[9]:


#Polynomial Regression of the 2nd degree.
y_10 = pol_regression(Data['x'], Data['y'], 10)#10th degree regression.
plt.plot(Data['x'], y_10, 'purple', label='2nd degree regression')
plt.plot(Data['x'], Data['y'], 'bo', label='True data')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("10th Degree Polynomial Regression")
plt.legend(loc='lower right')
plt.show()


# In[10]:


def get_weights(features_train, y_train, degree): #Function to compute polynomial coefficients.
    if(degree <= 0): # Error handling to catch an invalid degree
        degree = 1
    X = np.ones(features_train.shape) #Creates a premade array to increase computational speed.
    for i in range(1,degree+1): #To the nth degree.
       X = np.column_stack((X, features_train ** i)) #Creates an array of X values multiplied by power of i (to n).  
    XX = X.transpose().dot(X) #Creates a Vandermode matrix from X.
    w = np.linalg.solve(XX, X.transpose().dot(y_train)) #Creates polynomial weights.   
    return w # Returns weights alone.

def tt_regression(features_train, y_train, weights):
    X = np.ones(features_train.shape) #Creates a premade array to increase computational speed.
    # No need for error handling regarding degree, as that has already been handled in the function to calculate weights.
    for i in range(1,len(weights)): #To the nth degree.
       X = np.column_stack((X, features_train ** i)) #Creates an array of X values multiplied by power of i (to n).  
    pol_coefficients = X.dot(weights)  # Calculates predicted values
    return pol_coefficients # Returns polynomial coefficients

def eval_pol_regression(parameters, x, y): #Computes RMSE of each regression
    rmse = 0 #Sets to 0 by default
    for i in range(len(x)): #For every value in X
        rmse +=(parameters[i] - y[i]) ** 2 #Adds the predicted output minus the actual output squared to the RMSE 
    rmse = rmse / len(x) #Once that loop is finished, the final RMSE is calculated by dividing it by the input array size.
    return rmse #Returns the RMSE


# In[11]:


#Gets random values for training & test sets.
shuffleData = regData.iloc[:, 1:3] #Retrieves new X and Y.

shuffleData = shuffleData.sample(frac = 1) #Shuffles them.

trainData = shuffleData.iloc[0:12, 0:2] #Partitions the sets into a 70:30 split.
trainData.sort_values(by=['x'], inplace=True)
testData = shuffleData.iloc[12:20, 0:2]
testData.sort_values(by=['x'], inplace=True)

y_1_weights = get_weights(trainData['x'], trainData['y'], 1) # Gets the weights for all of the data.
y_2_weights = get_weights(trainData['x'], trainData['y'], 2)
y_3_weights = get_weights(trainData['x'], trainData['y'], 3)
y_6_weights = get_weights(trainData['x'], trainData['y'], 6)
y_10_weights = get_weights(trainData['x'], trainData['y'], 10)

y_1_test = tt_regression(testData['x'], testData['y'], y_1_weights) # Generates training and testing data from training weights.
y_2_test = tt_regression(testData['x'], testData['y'], y_2_weights)
y_3_test = tt_regression(testData['x'], testData['y'], y_3_weights)
y_6_test = tt_regression(testData['x'], testData['y'], y_6_weights)
y_10_test = tt_regression(testData['x'], testData['y'], y_10_weights)

y_1_train = tt_regression(trainData['x'], trainData['y'], y_1_weights) 
y_2_train = tt_regression(trainData['x'], trainData['y'], y_2_weights)
y_3_train = tt_regression(trainData['x'], trainData['y'], y_3_weights)
y_6_train = tt_regression(trainData['x'], trainData['y'], y_6_weights)
y_10_train = tt_regression(trainData['x'], trainData['y'], y_10_weights)


# In[12]:


trainY_1_eval = eval_pol_regression(y_1_train, trainData['x'].values, trainData['y'].values) # Evaluates the training sets.
trainY_2_eval = eval_pol_regression(y_2_train, trainData['x'].values, trainData['y'].values)
trainY_3_eval = eval_pol_regression(y_3_train, trainData['x'].values, trainData['y'].values)
trainY_6_eval = eval_pol_regression(y_6_train, trainData['x'].values, trainData['y'].values)
trainY_10_eval = eval_pol_regression(y_10_train, trainData['x'].values, trainData['y'].values)

testY_1_eval = eval_pol_regression(y_1_test, testData['x'].values, testData['y'].values) #  Evaluates the testing sets.
testY_2_eval = eval_pol_regression(y_2_test, testData['x'].values, testData['y'].values)
testY_3_eval = eval_pol_regression(y_3_test, testData['x'].values, testData['y'].values)
testY_6_eval = eval_pol_regression(y_6_test, testData['x'].values, testData['y'].values)
testY_10_eval = eval_pol_regression(y_10_test, testData['x'].values, testData['y'].values)



polNum = (1,2,3,6,10) # Number of polynomial
trainEvals = (trainY_1_eval, trainY_2_eval, trainY_3_eval, trainY_6_eval, trainY_10_eval) # Evaluation matrices
testEvals = (testY_1_eval, testY_2_eval, testY_3_eval, testY_6_eval, testY_10_eval)
plt.figure()
plt.semilogy(polNum, trainEvals, linestyle='--', marker='o') # Creates semiology of each evaluation matrix.
plt.semilogy(polNum, testEvals, linestyle='--', marker='o')
plt.xlabel('Degree of Polynomial')
plt.ylabel('Root Mean Squared Error')
plt.title('Root Mean Squared Error of Training and Test set (shuffled)')
plt.legend(('RMSE on training set', 'RMSE on test set'))

for x,y in zip(polNum,trainEvals): # For every value in the training/test set evaluation matrix.

    label = "{:.2f}".format(y) # Labels each data point with it's numeric value

    plt.annotate(label, # Plots text
                 (x,y), # At point of x, y
                 textcoords="offset points", # Text positioning
                 xytext=(8,-13), # Distance from text to point
                 ha='center') # Horizontal alignment

for x,y in zip(polNum, testEvals):

    label = "{:.2f}".format(y)

    plt.annotate(label, 
                 (x,y), 
                 textcoords="offset points", 
                 xytext=(-8,5), 
                 ha='center') 
    
plt.tight_layout(pad=-10) # Expands the border of the plot
plt.show()


# In[14]:


normData = regData.iloc[:, 1:3] #Retrieves unfiltered and unchanged data.

trainData = normData.iloc[0:12, 0:2] #Partitions the sets into a 70:30 split and sorts them after allocation.
trainData.sort_values(by=['x'], inplace=True)
testData = normData.iloc[12:20, 0:2]
testData.sort_values(by=['x'], inplace=True)
# Repeats the process of evaluating each data set, only this time with unshuffled data.
y_1_weights = get_weights(trainData['x'], trainData['y'], 1)
y_2_weights = get_weights(trainData['x'], trainData['y'], 2)
y_3_weights = get_weights(trainData['x'], trainData['y'], 3)
y_6_weights = get_weights(trainData['x'], trainData['y'], 6)
y_10_weights = get_weights(trainData['x'], trainData['y'], 10)

y_1_test = tt_regression(testData['x'], testData['y'], y_1_weights)
y_2_test = tt_regression(testData['x'], testData['y'], y_2_weights)
y_3_test = tt_regression(testData['x'], testData['y'], y_3_weights)
y_6_test = tt_regression(testData['x'], testData['y'], y_6_weights)
y_10_test = tt_regression(testData['x'], testData['y'], y_10_weights)

y_1_train = tt_regression(trainData['x'], trainData['y'], y_1_weights)
y_2_train = tt_regression(trainData['x'], trainData['y'], y_2_weights)
y_3_train = tt_regression(trainData['x'], trainData['y'], y_3_weights)
y_6_train = tt_regression(trainData['x'], trainData['y'], y_6_weights)
y_10_train = tt_regression(trainData['x'], trainData['y'], y_10_weights)


# In[15]:


trainY_1_eval = eval_pol_regression(y_1_train, trainData['x'].values, trainData['y'].values)
trainY_2_eval = eval_pol_regression(y_2_train, trainData['x'].values, trainData['y'].values)
trainY_3_eval = eval_pol_regression(y_3_train, trainData['x'].values, trainData['y'].values)
trainY_6_eval = eval_pol_regression(y_6_train, trainData['x'].values, trainData['y'].values)
trainY_10_eval = eval_pol_regression(y_10_train, trainData['x'].values, trainData['y'].values)

testY_1_eval = eval_pol_regression(y_1_test, testData['x'].values, testData['y'].values)
testY_2_eval = eval_pol_regression(y_2_test, testData['x'].values, testData['y'].values)
testY_3_eval = eval_pol_regression(y_3_test, testData['x'].values, testData['y'].values)
testY_6_eval = eval_pol_regression(y_6_test, testData['x'].values, testData['y'].values)
testY_10_eval = eval_pol_regression(y_10_test, testData['x'].values, testData['y'].values)

# Creates a semiology for the unshuffled matrices.

polNum = (1,2,3,6,10)
trainEvals = (trainY_1_eval, trainY_2_eval, trainY_3_eval, trainY_6_eval, trainY_10_eval)
testEvals = (testY_1_eval, testY_2_eval, testY_3_eval, testY_6_eval, testY_10_eval)
plt.figure()
plt.semilogy(polNum, trainEvals, linestyle='--', marker='o')
plt.semilogy(polNum, testEvals, linestyle='--', marker='o')
plt.xlabel('Degree of Polynomial')
plt.ylabel('Root Mean Squared Error')
plt.title('Root Mean Squared Error of Training and Test set (True)')
plt.legend(('RMSE on training set', 'RMSE on test set'))

for x,y in zip(polNum,trainEvals):

    label = "{:.2f}".format(y)

    plt.annotate(label, 
                 (x,y), 
                 textcoords="offset points",
                 xytext=(-5,-17),
                 ha='center') 

for x,y in zip(polNum, testEvals):

    label = "{:.2f}".format(y)

    plt.annotate(label,
                 (x,y), 
                 textcoords="offset points", 
                 xytext=(8,8), 
                 ha='center') 
    
plt.tight_layout(pad=-10)
plt.show()


# In[32]:


# TASK 2
# Importing additional modules for K-Means clustering. Some from task 1 are still required.
import random as rd
import copy 
import math
KMeansData = pd.read_csv("Task2Data.csv") # Reads the data (replace file path)


# In[36]:


def compute_distance(vec_1, vec_2): # Calculates Euclidean distance from one vector to another
    distance = norm(vec_1 - vec_2) # Returns the Euclidean Distance 
    #distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])) # <- Alternative method of calculating distance. 
    # It produces the exact same results as calling np.norm().
    return distance # Returns distance

def init_centroids(dataset, k): # Uses random data samples to initialise centroids.
    centroids = dataset.sample(n=k) # Gets k samples from the dataset and declares them as centroids
    return centroids # Returns centroids
    
def kmeans(dataset, k): # Function for k-means clustering
    name = ["Centroid %d" % i for i in range(k)] # Name equal to number of Centroids ('Centroid 0 ... Centroid k)
    colmap = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'm', 5: 'c'} # Up to five possible colours for clusters.
    centroids = init_centroids(dataset, k) #Initialises centroids
    
    for i in range(k): #For every cluster
        dataset[name[i]] = dataset.apply(lambda x : compute_distance(x.values[0:centroids.shape[1]], centroids.iloc[i].values), axis=1)
        # Calculates the Euclidean distance for every data feature to every centroid, storing the computed values in the dataset.
    calls_to_use = list(filter(lambda f:'Centroid' in f, dataset.columns)) # For every column labeled with 'Centroid' in the dataset
    dataset['Closest'] = dataset[calls_to_use].idxmin(axis=1) # Gets the closest centroid 
    dataset['Closest'] = dataset['Closest'].map(lambda x: int(x.lstrip('Centroid '))) # Gets the centroid number
    dataset['Closest Value'] = dataset[calls_to_use].min(axis=1) # Gets the closest value
    
    hardstop = 0 # Hardstop coded to stop infinite-looping
    while True: # While loop
        iteration = ["Iteration %d" % i for i in range(hardstop)] #Iteration number increases 
        closest_centroids = dataset['Closest'].copy(deep=True) # Closest centroids are stored
        for j in range(k): # For every centroid
            # Calculate the mean of every cluster and assign a new centroid for each
            centroids.iloc[j][0] = np.mean(dataset[dataset['Closest'] == j]['height'])
            centroids.iloc[j][1] = np.mean(dataset[dataset['Closest'] == j]['tail length'])
            centroids.iloc[j][2] = np.mean(dataset[dataset['Closest'] == j]['leg length'])
            centroids.iloc[j][3] = np.mean(dataset[dataset['Closest'] == j]['nose circumference'])
            
        for i in range(k): #Calculates Euclidean Distance again
            dataset[name[i]] = dataset.apply(lambda x : compute_distance(x.values[0:centroids.shape[1]], centroids.iloc[i].values), axis=1)
             
        # Repeats assignment of closest centroid & value
        dataset['Closest'] = dataset[calls_to_use].idxmin(axis=1)
        dataset['Closest'] = dataset['Closest'].map(lambda x: int(x.lstrip('Centroid '))) 
        dataset['Closest Value'] = dataset[calls_to_use].min(axis=1)
        
        # Gets SSE for each iteration
        dataset[f'Iteration{hardstop}'] = sum((dataset['Closest Value'] - np.mean(dataset['Closest Value']))**2)
        if closest_centroids.equals(dataset['Closest']): # If there are no changes in centroids (the centroids are already the mean)
            print(f'Found {k} clusters in {hardstop + 1} iterations') # Performance check
            break # The while loop breaks
        else: # If the centroids do change
            hardstop += 1 # Hardstop increases by one
            if(hardstop == 60): # And stops the loop once it reaches a set boundary
                print(f'Stopped upon iteration {hardstop}')
                break
    dataset['Iterations needed'] = hardstop + 1 # Total iterations equals hardstop + 1 (as it starts from 0)
    dataset['Colour'] = dataset['Closest'].map(lambda x: colmap[x])  # Maps the cluster colours to the dataset
    return centroids, dataset # Returns calculated K-Means


# In[38]:


cluster_2_input = KMeansData[["height", "tail length", "leg length", "nose circumference"]] # Input for 2 clusters.
cluster_3_input = KMeansData[["height", "tail length", "leg length", "nose circumference"]] # Input for 3 clusters.
centroids_2, data_2 = kmeans(cluster_2_input, 2) # Computes K-Means clustering for each input
centroids_3, data_3 = kmeans(cluster_3_input, 3)
# Two inputs are required as each one is modified otherwise.


# In[35]:


# Scatters the data and each centroid. The centroids are black to highlight their positions.
plt.scatter(data_2['height'], data_2['tail length'], c = data_2['Colour'], alpha = 0.4, marker = "*", label='Data')
plt.scatter(centroids_2["height"], centroids_2["tail length"], c = 'black' ,marker = "o", label='Centroid', edgecolors='black')
plt.xlabel("Height")
plt.ylabel("Tail Length")
plt.title("Clustering using Height and Tail Length (2 Centroids)")
plt.legend(loc='lower right')
plt.show()


# In[24]:


plt.scatter(data_2['height'], data_2['leg length'], c = data_2['Colour'], alpha = 0.4, marker = "*", label='Data')
plt.scatter(centroids_2["height"], centroids_2["leg length"], c = 'black' ,marker = "o", label='Centroid', edgecolors='black')
plt.xlabel("Height")
plt.ylabel("Leg Length")
plt.title("Clustering using Height and Leg Length (2 Centroids)")
plt.legend(loc='lower right')
plt.show()


# In[25]:


plt.scatter(data_3['height'], data_3['tail length'], c = data_3['Colour'], alpha = 0.4, marker = "*", label='Data')
plt.scatter(centroids_3["height"], centroids_3["tail length"], c = 'black' ,marker = "o", label='Centroid', edgecolors='black')
plt.xlabel("Height")
plt.ylabel("Tail Length")
plt.title("Clustering using Height and Tail Length (3 Centroids)")
plt.legend(loc='lower right')
plt.show()


# In[26]:


plt.scatter(data_3['height'], data_3['leg length'], c = data_3['Colour'], alpha = 0.4, marker = "*", label='Data')
plt.scatter(centroids_3["height"], centroids_3["leg length"], c = 'black' ,marker = "o", label='Centroid', edgecolors='black')
plt.xlabel("Height")
plt.ylabel("Leg Length")
plt.title("Clustering using Height and Leg Length (3 Centroids)")
plt.legend(loc='lower right')
plt.show()


# In[27]:


# Plot the SSE for K clusters. 
it_number_2 = data_2['Iterations needed'][0] # The number of iterations is retrieved (x)
it_2 = ["Iteration%d" % i for i in range(0, it_number_2)] # Labels it.

err_graph_2 = [] # Creates empty graph.
for l in range(0, it_number_2): # For every iteration
    err_graph_2 = np.append(err_graph_2, data_2[it_2[l]].values.min()) # Plot the iteration number and value as a line graph.

plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("SSE of K-Means Clustering (2 Centroids)")
plt.plot(range(it_number_2), err_graph_2)
plt.show()


# In[28]:


it_number_3 = data_3['Iterations needed'][0]
it_3 = ["Iteration%d" % i for i in range(0, it_number_3)]

err_graph_3 = []
for l in range(0, it_number_3):
    err_graph_3 = np.append(err_graph_3, data_3[it_3[l]].values.min())

plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("SSE of K-Means Clustering (3 Centroids)")
plt.plot(range(it_number_3), err_graph_3)
plt.show()

