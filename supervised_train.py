#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: supervised model training

Usage:

    $ spark-submit supervised_train.py hdfs:/path/to/file.parquet hdfs:/path/to/save/model

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main(spark, data_file, model_file):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    ###

    # Read data
    df = spark.read.parquet(data_file)

    # Take 1/10 data without replacement
    df = df.sample(False, 0.1, seed = 0)

    # Vectorize selected features
    features = ['mfcc_' + '%.2d' % i for i in range(20)]
    assembler = VectorAssembler(inputCols=features, outputCol="vectorized_features")

    # Standardize the features
    scaler = StandardScaler(inputCol="vectorized_features", outputCol="scaled_features", withStd=True, withMean=False)

    # Transform string target variable into numerical
    indexer = StringIndexer(inputCol="genre", outputCol="label", handleInvalid = "skip")

    # Build logistic regression
    lr = LogisticRegression(maxIter=20, featuresCol = scaler.getOutputCol(), labelCol=indexer.getOutputCol())

    # Build a pipeline
    pipeline = Pipeline(stages = [assembler, scaler, indexer, lr])

    # Build parameter grid and cross validation
    paramGrid = ParamGridBuilder().addGrid(lr.elasticNetParam,[0.1,0.3,0.5,0.8]).addGrid(lr.regParam, [0.1,0.08,0.05,0.02,0.01]).build()

    crossval = CrossValidator(estimator = pipeline, estimatorParamMaps = paramGrid, evaluator = MulticlassClassificationEvaluator(), numFolds = 5)

    # Save model
    cvModel = crossval.fit(df)
    cvModel.bestModel.write().overwrite().save(model_file)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)
