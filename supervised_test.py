#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: supervised model testing

Usage:

    $ spark-submit supervised_test.py hdfs:/path/to/load/model.parquet hdfs:/path/to/file

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import PipelineModel

def main(spark, model_file, data_file):
    '''Main routine for supervised evaluation

    Parameters
    ----------
    spark : SparkSession object

    model_file : string, path to store the serialized model file

    data_file : string, path to the parquet file to load
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    # Overall precision, recall, and F1
    # Weighted precision, recall, and F1
    # Per-class precision, recall, and F1
    ###

    # Load the model
    model = PipelineModel.load(model_file)
    print("Best Model Parameters: ", model.stages[-1].extractParamMap())

    # Read the val
    val = spark.read.parquet(data_file)
    
    # Make predictions on val
    predictions = model.transform(val)
    
    # Compute raws scores on val
    predictionAndLabels = predictions.select("prediction","label")
    predlabels = predictionAndLabels.rdd.map(tuple)

    # Instantiate metrics object
    metrics = MulticlassMetrics(predlabels)

    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Overall Statistics Summary")
    print("precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)

    # Weighted statistics
    print("-----------------------------")
    print("Weighted recall = %s" % metrics.weightedRecall)
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    #print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    #print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)

    # Preclass Statistics
    print("-----------------------------")
    labels = predictions.select("label").distinct().rdd.map(lambda lp: lp[0]).collect()
    for label in sorted(labels):
        print("Class %s precision = %s" % (label, metrics.precision(label)))
        print("Class %s recall = %s" % (label, metrics.recall(label)))
        print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))





# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_test').getOrCreate()

    # And the location to store the trained model
    model_file = sys.argv[1]

    # Get the filename from the command line
    data_file = sys.argv[2]

    # Call our main routine
    main(spark, model_file, data_file)
