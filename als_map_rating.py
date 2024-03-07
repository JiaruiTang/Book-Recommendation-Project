#!/usr/bin/env python
import math
import datetime
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql import functions as F 
from pyspark.sql.window import Window
from pyspark.sql.functions import when

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

def main(spark):
    print(spark.sparkContext.getConf().getAll())

    train = spark.read.parquet('hdfs:/user/cl5293/recommend/2.0_train.parquet')
    test = spark.read.parquet('hdfs:/user/cl5293/recommend/2.0_test.parquet')
    df = spark.read.parquet('hdfs:/user/cl5293/recommend/2.0_valid_test.parquet')

    train=train.withColumn('rating', train.rating - 3)
    test=test.withColumn('rating', test.rating - 3)
    df=df.withColumn('rating', df.rating - 3)


    perUserActualItemsDF = df.withColumn('rating', when(df.rating<3, None).otherwise(df.rating))\
        .groupBy("user_id").agg(F.collect_list("book_id").alias("Actual"))

    als = ALS(rank=20, maxIter=10, regParam=0.1,
        userCol='user_id',
        itemCol='book_id',
        ratingCol='rating',
        coldStartStrategy="drop",
        nonnegative=True
        )
    model = als.fit(train)
    user_subset_recs = model.recommendForUserSubset(test.select('user_id'), 500)
    perUserPredictItemsDF = (user_subset_recs.groupBy('user_id')
        .agg(F.expr('collect_list(recommendations.book_id)[0] as books')))

    perUserItemsRDD = perUserPredictItemsDF.join(perUserActualItemsDF, 'user_id','left').repartition(1000) \
    .rdd \
    .map(lambda row: (row[1], row[2]))
    rankingMetrics = RankingMetrics(perUserItemsRDD)

    print('mean Average Precision: ',rankingMetrics.meanAveragePrecision)
    print('precision At (500): ',rankingMetrics.precisionAt(500))
    print('ndcg At (500): ',rankingMetrics.ndcgAt(500),'\n')

    predictions = model.transform(test)
    windowSpec = Window.partitionBy('user_id').orderBy(F.col('prediction').desc())
    perUserPredictItemsDF = (predictions
            .select('user_id', 'book_id', 'prediction', F.rank().over(windowSpec).alias('rank'))
            .where('rank <= 500 and rating>=3')
            .groupBy('user_id')
            .agg(F.expr('collect_list(book_id) as books')))


    perUserItemsRDD = perUserPredictItemsDF.join(perUserActualItemsDF, 'user_id','left').repartition(1000) \
    .rdd \
    .map(lambda row: (row[1], row[2]))
    rankingMetrics = RankingMetrics(perUserItemsRDD)

    print('mean Average Precision: ',rankingMetrics.meanAveragePrecision)
    print('precision At (500): ',rankingMetrics.precisionAt(500))
    print('ndcg At (500): ',rankingMetrics.ndcgAt(500))


if __name__ == "__main__":
    memory = '5g'
    spark = (SparkSession.builder
             .appName('train_als')
             .master('local[*]')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    main(spark)
    print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    spark.stop()
