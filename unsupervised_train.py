#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: unsupervised model training

Usage:

    $ spark-submit unsupervised_train.py hdfs:/path/to/file.parquet hdfs:/path/to/save/model

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

def main(spark, data_file, model_file):
    '''Main routine for unsupervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    # 1) Select out the 20 attribute columns labeled mfcc_00, mfcc_01, ..., mfcc_19
    # 2) Normalize the features using a StandardScaler
    # 3) Fit a K-means clustering model to the standardized data with K=100.
    ###

    # Read the data
    df = spark.read.parquet(data_file)

    # Vectorize features
    features = ['mfcc_' + '%.2d' % i for i in range(20)]
    assembler = VectorAssembler(inputCols = features, outputCol = "vectorized_features")

    # Standardize the features
    scaler = StandardScaler(inputCol = "vectorized_features", outputCol = "scaled_features", withStd = True, withMean = False)

    # Fit K-means
    kmeans100 = KMeans(featuresCol = "scaled_features", k = 100, seed = 1)

    # Build a pipeline
    pipeline = Pipeline(stages = [assembler, scaler, kmeans100])
    kmeans100_model = pipeline.fit(df)

    # Save the model
    kmeans100_model.save(model_file)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('unsupervised_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)
