from pyspark.sql import SparkSession

from pyspark.sql import functions as F 
from pyspark.sql.window import Window
from pyspark.sql.functions import when

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import StringIndexer

book_join1 = spark.read.csv('hdfs:/user/jt3869/book_join1.csv')
book_join1 = book_join1.fillna( { '_c154':'na', '_c155':0, '_c156':0, '_c157':'na', '_c158': 'na'} )

# convert categorical data
col = ['_c153','_c154','_c155','_c156','_c157','_c158']
book_category = book_join1
for i in col:
    string = i+'i'
    indexer = StringIndexer(inputCol=i, outputCol=string)
    book_category = indexer.fit(book_category).transform(book_category).drop(i)

book_category.write.csv('hdfs:/user/jt3869/book_category.csv')

feature_list = book_category.columns[152:159]

label_list = book_category.columns[1:151]

# Leave out subset data for evaluation later
(trainingData, testData) = book_category.randomSplit([0.8, 0.2])

clfs = []
test_prediction = np.zeros([testData.shape[0], 150])
train_prediction = np.zeros([trainingData.shape[0], 150])
for i in range(150):
    new_feature_list = feature_list + label_list[:i]
    assembler = VectorAssembler(inputCols=new_feature_list, outputCol="features")
    rf = RandomForestRegressor(labelCol=label_list[i], featuresCol="features")
    pipeline = Pipeline(stages=[assembler, rf])
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 50, num = 3)]) \
        .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 3)]) \
        .build()
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(),
                              numFolds=3)
    
    clf = crossval.fit(trainingData)
    clfs.append(clf)
    train_prediction[:, i] = clf.transform(trainingData)
    test_prediction[:, i] = clf.transform(testData)

















