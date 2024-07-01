import pandas
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
import sklearn.grid_search
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize

#Reads in an input csv file and converts it into a pandas dataframe.Returns the encoded labels and the dataframe.
#So, how this works is you make a dataframe, manipulate the columns and encode stuff and add it back so every feature is a number
def preProcessDataset():
    f= open('EnglishMoviesWithPopularityFinal.csv')
    movieFrame = pandas.read_csv(f)

    intRuntime=[]

    objRuntime = movieFrame['Runtime'].tolist()
    for item in objRuntime:
        intRuntime.append(int(item.split(' ')[0]))
        movieFrame['runtime'] = pandas.Series(intRuntime)

    ratingList = movieFrame['Rated'].tolist()
    genreList = movieFrame['genre'].tolist()
    tomatoImageList = movieFrame['tomatoImage'].tolist()

    labelEncoderRating = LabelEncoder()
    labelEncoderGenre = LabelEncoder()
    labelEncoderTomatoImage = LabelEncoder()
        
    labelEncoderRating.fit(ratingList)
    labelsRating = labelEncoderRating.transform(ratingList)
    movieFrame['rating']=pandas.Series(labelsRating)

    labelEncoderGenre.fit(genreList)
    genreLabels = labelEncoderGenre.transform(genreList)
    movieFrame['Genre'] = pandas.Series(genreLabels)

    labelEncoderTomatoImage.fit(tomatoImageList)
    tomatoLabels = labelEncoderTomatoImage.transform(tomatoImageList)
    movieFrame['tomatoImages']=tomatoLabels

    movieFrame=movieFrame.drop(['Title','Director','Released','Runtime','Actors','Rated','genre','tomatoImage'],1)
    return labelEncoderRating,labelEncoderGenre,labelEncoderTomatoImage,movieFrame

#Takes in the movieframe, stores the headers so that you don't lose track of what's what. Later converts the dataframe into a numpy array for skearn to do its magic
def prepareDataset(movieFrame):

    headers = list(movieFrame)


    YIndex = headers.index('revenue')
    datasetMatrix = movieFrame.as_matrix()

    datasetTrain = datasetMatrix[0:1500]
    datasetTrainWithoutLabels = np.delete(datasetTrain,YIndex,1)

    labels = datasetTrain[:,YIndex]

    datasetTest = datasetMatrix[1500:datasetMatrix.shape[0]]
    datasetTestWithoutLabels = np.delete(datasetTest,YIndex,1)

    trueLabels = datasetTest[:,YIndex]
    return headers,datasetMatrix,datasetTrainWithoutLabels,labels,datasetTest,datasetTestWithoutLabels,trueLabels

#Same stuff as above, but takes inflation into account
def prepareInflationAdjustedDataset(movieFrame):
    
    cpiDict = {2000:172.2,
            2001:   177.1,
            2002:   179.9,
            2003:   184,
            2004:   188.9,
            2005:   195.3,
            2006:   201.6,
            2007:   207.3,
            2008:   215.303,
            2009:   214.537,
            2010:   218.056,
            2011:   224.939,
            2012:   229.594,
            2013:   232.957,
            2014:   236.736,
            2015:   237.017}
    budgetList = movieFrame['budgets'].tolist()
    revenueList = movieFrame['revenue'].tolist()
    yearList = movieFrame['Year'].tolist()
    
    inflationAdjustedBudgetList=[]
    inflationAdjustedRevenueList=[]
    for i in range(len(budgetList)):
        inflationAdjustedBudget = (float)(237.017)/cpiDict[yearList[i]]*budgetList[i]
        inflationAdjustedRevenue = (float)(237.017)/cpiDict[yearList[i]]*revenueList[i]
        inflationAdjustedBudgetList.append(inflationAdjustedBudget)
        inflationAdjustedRevenueList.append(inflationAdjustedRevenue)
   
    movieFrame['inflationBudget'] = pandas.Series(inflationAdjustedBudgetList)
    movieFrame['inflationRevenue'] = pandas.Series(inflationAdjustedRevenueList)

    movieFrame = movieFrame.drop(['budgets','revenue'],1)
    headers = list(movieFrame)
    
    
    YIndex = headers.index('inflationRevenue')
    datasetMatrix = movieFrame.as_matrix()

    datasetTrain = datasetMatrix[0:1500]
    datasetTrainWithoutLabels = np.delete(datasetTrain,YIndex,1)

    labels = datasetTrain[:,YIndex]

    datasetTest = datasetMatrix[1500:datasetMatrix.shape[0]]
    datasetTestWithoutLabels = np.delete(datasetTest,YIndex,1)

    trueLabels = datasetTest[:,YIndex]
    return headers,datasetMatrix,datasetTrainWithoutLabels,labels,datasetTest,datasetTestWithoutLabels,trueLabels

#Everything from here is functions that train and test stuff 
def ridgePredictions(datasetTrainWithoutLabels,labels,datasetTestWithoutLabels,trueLabels):
    regressor = Ridge()
    featureArray = np.arange(1,datasetTrainWithoutLabels.shape[1])
    alphas=np.arange(1,50)
    steps=[('regressor',regressor)]
    pipeline=Pipeline(steps)
    parameterGrid = dict(regressor__alpha = alphas)
    GridSearchResult = sklearn.grid_search.GridSearchCV(pipeline,param_grid=parameterGrid,cv = 5)

    GridSearchResult.fit(datasetTrainWithoutLabels,labels)
    a =  GridSearchResult.grid_scores_
    scores=[]
    for t in a:
        scores.append(t.mean_validation_score)
    plt.plot(alphas,scores)
    plt.grid(True)
    plt.ylabel('Mean Validation Score')
    plt.xlabel('Alphas')
    plt.title('Mean Validation Score for Ridge Regression')
    plt.tight_layout()
    plt.savefig('Ridge Validation Score')
    plt.show()
    predictions = GridSearchResult.predict(datasetTestWithoutLabels)
    print r2_score(trueLabels,predictions)
    return predictions

def lassoPredictions(datasetTrainWithoutLabels,labels,datasetTestWithoutLabels,trueLabels):
    regressor = Lasso()
    featureArray = np.arange(1,datasetTrainWithoutLabels.shape[1])
    alphas=np.arange(1,50)
    steps=[('regressor',regressor)]
    pipeline=Pipeline(steps)
    parameterGrid = dict(regressor__alpha = alphas)
    GridSearchResult = sklearn.grid_search.GridSearchCV(pipeline,param_grid=parameterGrid,cv = 5)
    GridSearchResult.fit(datasetTrainWithoutLabels,labels)
    a =  GridSearchResult.grid_scores_
    scores=[]
    for t in a:
        scores.append(t.mean_validation_score)
    plt.plot(alphas,scores)
    plt.grid(True)
    plt.ylabel('Mean Validation Score')
    plt.xlabel('Alphas')
    plt.title('Mean Validation Score for Lasso Regression')
    plt.tight_layout()
    plt.savefig('Lasso Validation Score')
    plt.show()
    
    print GridSearchResult.best_params_
    predictions = GridSearchResult.predict(datasetTestWithoutLabels)
    print r2_score(trueLabels,predictions)
    return predictions

def decisionTreePredictions(datasetTrainWithoutLabels,labels,datasetTestWithoutLabels,trueLabels):
    regressor = DecisionTreeRegressor()
    #featureArray = np.arange(1,datasetTrainWithoutLabels.shape[1])
    maxDepths = np.arange(1,datasetTestWithoutLabels.shape[1])
    steps=[('regressor',regressor)]
    pipeline=Pipeline(steps)
    parameterGrid = dict(regressor__max_depth = maxDepths)
    GridSearchResult = sklearn.grid_search.GridSearchCV(pipeline,param_grid=parameterGrid,cv = 5)
        
    GridSearchResult.fit(datasetTrainWithoutLabels,labels)
    a =  GridSearchResult.grid_scores_
    scores=[]
    for t in a:
        scores.append(t.mean_validation_score)
    plt.plot(maxDepths,scores)
    plt.grid(True)
    plt.ylabel('Mean Validation Score')
    plt.xlabel('Max Depth')
    plt.title('Mean Validation Score for Decision Tree Regression')
    plt.tight_layout()
    plt.savefig('Decision Tree Validation Score')
    plt.show()
    
    print GridSearchResult.best_params_
    predictions = GridSearchResult.predict(datasetTestWithoutLabels)
    print r2_score(trueLabels,predictions)
    return predictions,GridSearchResult


def AdaBoostRegressorPredictions(datasetTrainWithoutLabels,labels,datasetTestWithoutLabels,trueLabels,bestDTree):
    
    regressor = AdaBoostRegressor(base_estimator = bestDTree,n_estimators = 100)

    featureArray = np.arange(1,datasetTrainWithoutLabels.shape[1])
    regressor.fit(datasetTrainWithoutLabels,labels)
    predictions = regressor.predict(datasetTestWithoutLabels)
    
    print r2_score(trueLabels,predictions)
    return predictions

def plotFeatureVariation(dataset,headers):
    xArray = dataset[:,headers.index('sentiment_polarity')]
    yArray = dataset[:,headers.index('revenue')]
    plt.scatter(xArray,yArray)
    plt.xlabel('Sentiment polarity of the movie')
    plt.ylabel('Revenue')
    
    plt.title("Influence of sentiment polarity on the revenue of a movie")
    plt.tight_layout()
    plt.savefig('SentimentScorevRevenue.png')
    plt.show()
    
#Write this function that labels the revenue and returns a "classification-ready" numpy array.
#def prepareDatasetForClassification(dataset):
    

'''
def svrPredictions(datasetTrainWithoutLabels,labels,datasetTestWithoutLabels,trueLabels):
    regressor = SVR(C=1000000000,epsilon=0.5)
    regressor.fit(datasetTrainWithoutLabels,labels)
    predictions = regressor.predict(datasetTestWithoutLabels)
    return predictions
'''

#Calling all the above functions.
labelEncoderRating,labelEncoderGenre,labelEncoderTomatoImage,movieFrame = preProcessDataset()
headers,dataset,datasetTrainWithoutLabels,labels,datasetTest,datasetTestWithoutLabels,trueLabels = prepareDataset(movieFrame)
plotFeatureVariation(dataset,headers)

headersIA,datasetIA,datasetTrainWithoutLabelsIA,labelsIA,datasetTestIA,datasetTestWithoutLabelsIA,trueLabelsIA = prepareInflationAdjustedDataset(movieFrame)

RidgeY = ridgePredictions(datasetTrainWithoutLabels,labels,datasetTestWithoutLabels,trueLabels)
LassoY = lassoPredictions(datasetTrainWithoutLabels,labels,datasetTestWithoutLabels,trueLabels)
DTreeY,bestDTree = decisionTreePredictions(datasetTrainWithoutLabels,labels,datasetTestWithoutLabels,trueLabels)
AdaBoostY = AdaBoostRegressorPredictions(datasetTrainWithoutLabels,labels,datasetTestWithoutLabels,trueLabels,bestDTree)



print '\n'

RidgeYIA = ridgePredictions(datasetTrainWithoutLabelsIA,labelsIA,datasetTestWithoutLabelsIA,trueLabelsIA)
LassoYIA = lassoPredictions(datasetTrainWithoutLabelsIA,labelsIA,datasetTestWithoutLabelsIA,trueLabelsIA)
DTreeYIA,bestDTreeIA = decisionTreePredictions(datasetTrainWithoutLabelsIA,labelsIA,datasetTestWithoutLabelsIA,trueLabelsIA)
AdaBoostYIA = AdaBoostRegressorPredictions(datasetTrainWithoutLabelsIA,labelsIA,datasetTestWithoutLabelsIA,trueLabelsIA,bestDTree)

