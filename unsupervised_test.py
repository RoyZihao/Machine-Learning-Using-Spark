#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: unsupervised model testing

Usage:

    $ spark-submit unsupervised_test.py hdfs:/path/to/load/model.parquet hdfs:/path/to/file

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import PipelineModel

def main(spark, model_file, data_file):
    '''Main routine for unsupervised evaluation

    Parameters
    ----------
    spark : SparkSession object

    model_file : string, path to store the serialized model file

    data_file : string, path to the parquet file to load
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    ###

    # Load the Pipeline model we trained
    model = PipelineModel.load(model_file)

    # Read the val
    val = spark.read.parquet(data_file)

    # Predictions
    predictions = model.transform(val)

    # Evaluations
    evaluator = ClusteringEvaluator(predictionCol = 'prediction',featuresCol = 'scaled_features')
    result = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(result))





# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('unsupervised_test').getOrCreate()

    # And the location to store the trained model
    model_file = sys.argv[1]

    # Get the filename from the command line
    data_file = sys.argv[2]

    # Call our main routine
    main(spark, model_file, data_file)
