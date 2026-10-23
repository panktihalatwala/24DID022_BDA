import os
import time
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF

spark = (
    SparkSession.builder
    .appName("FullBatch_vs_MiniBatch_GD")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("escape", '"')
    .csv("hdfs:///user/student/sms_spam_large.csv")
    .select("label", "message_text")
    .dropna()
    .filter(col("label").isin("ham", "spam"))
    .filter(col("message_text") != "")
    .withColumn(
        "label",
        when(col("label") == "spam", 1.0).otherwise(0.0)
    )
)

train, test = df.randomSplit([0.8, 0.2], seed=42)

tokenizer = RegexTokenizer(
    inputCol="message_text", outputCol="tokens",
    pattern="\\W+", minTokenLength=2
)
stopwords = StopWordsRemover(
    inputCol="tokens", outputCol="filtered_tokens"
)
tf = HashingTF(
    inputCol="filtered_tokens", outputCol="raw_features",
    numFeatures=1 << 12
)
idf = IDF(
    inputCol="raw_features", outputCol="features", minDocFreq=2
)

feature_pipeline = Pipeline(stages=[tokenizer, stopwords, tf, idf])
feature_model = feature_pipeline.fit(train)

train_rdd = (
    feature_model.transform(train)
    .select("label", "features")
    .rdd.map(lambda r: (float(r.label), r.features))
    .cache()
)
test_rdd = (
    feature_model.transform(test)
    .select("label", "features")
    .rdd.map(lambda r: (float(r.label), r.features))
    .cache()
)

train_count = train_rdd.count()
test_count = test_rdd.count()
dimension = 1 << 12
learning_rate = 0.5
regularization = 0.001
epochs = 5


def sigmoid(value):
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + np.exp(-value))


def calculate_gradient(data_rdd, weights):
    """Aggregate partial gradients computed by Spark partitions."""
    def partition_gradient(iterator):
        gradient = np.zeros(dimension, dtype=np.float64)
        count = 0

        for label, features in iterator:
            indices = features.indices
            values = features.values

            score = np.dot(weights[indices], values)
            error = sigmoid(score) - label
            gradient[indices] += error * values
            count += 1

        if count:
            yield gradient, count

    partials = data_rdd.mapPartitions(partition_gradient).collect()

    total_gradient = np.zeros(dimension, dtype=np.float64)
    total_count = 0

    for gradient, count in partials:
        total_gradient += gradient
        total_count += count

    if total_count:
        total_gradient = total_gradient / total_count
        total_gradient += regularization * weights

    return total_gradient, total_count


def train_model(use_minibatch):
    weights = np.zeros(dimension, dtype=np.float64)
    start = time.perf_counter()

    for epoch in range(epochs):
        if use_minibatch:
            batch = train_rdd.sample(
                withReplacement=False,
                fraction=0.20,
                seed=100 + epoch
            )
            batch_count = batch.count()

            if batch_count == 0:
                batch = train_rdd
        else:
            batch = train_rdd

        gradient, used_count = calculate_gradient(batch, weights)
        weights -= learning_rate * gradient

        print(
            ("Mini-batch" if use_minibatch else "Full-batch"),
            "epoch", epoch + 1,
            "records used:", used_count
        )

    elapsed = time.perf_counter() - start
    return weights, elapsed


def evaluate_model(weights):
    def classify(record):
        label, features = record
        score = np.dot(weights[features.indices], features.values)
        prediction = 1.0 if sigmoid(score) >= 0.5 else 0.0
        return (1 if prediction == label else 0, 1)

    counts = test_rdd.map(classify).reduce(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )
    return counts[0] / counts[1]


print("\n========== FULL-BATCH GRADIENT DESCENT ==========")
full_weights, full_time = train_model(use_minibatch=False)
full_accuracy = evaluate_model(full_weights)

print("\n========== MINI-BATCH GRADIENT DESCENT ==========")
mini_weights, mini_time = train_model(use_minibatch=True)
mini_accuracy = evaluate_model(mini_weights)

print("\n========== OPTIMIZATION COMPARISON ==========")
print("Training records:", train_count)
print("Testing records:", test_count)
print("Full-batch time:", round(full_time, 4), "seconds")
print("Mini-batch time:", round(mini_time, 4), "seconds")
print("Full-batch accuracy:", round(full_accuracy, 4))
print("Mini-batch accuracy:", round(mini_accuracy, 4))

os.makedirs("results", exist_ok=True)

with open("results/optimization_comparison.csv", "w") as f:
    f.write("Method,Epochs,TrainingSeconds,TestAccuracy\n")
    f.write(f"FullBatch,{epochs},{full_time},{full_accuracy}\n")
    f.write(f"MiniBatch20Percent,{epochs},{mini_time},{mini_accuracy}\n")

train_rdd.unpersist()
test_rdd.unpersist()
spark.stop()

print("Saved results/optimization_comparison.csv")
