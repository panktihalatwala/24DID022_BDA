import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

spark = (
    SparkSession.builder
    .appName("SecureComm_SMS_Spam_Classification")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
os.makedirs("results", exist_ok=True)

print("\n========== PRACTICAL 11: SMS SPAM CLASSIFICATION ==========\n")

# TASK 1: Load data from HDFS
df = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("escape", '"')
    .csv("hdfs:///user/student/sms_spam_large.csv")
    .select("label", "message_text")
    .dropna()
    .filter(col("label").isin("ham", "spam"))
    .withColumn(
        "label",
        when(col("label") == "spam", 1.0).otherwise(0.0)
    )
)

df = df.filter(col("message_text") != "")

print("========== TASK 1: DATASET ==========")
print("Total records:", df.count())
df.show(5, truncate=60)

# Split before fitting the text-processing pipeline.
train, test = df.randomSplit([0.8, 0.2], seed=42)
train.cache()
test.cache()

print("Training records:", train.count())
print("Testing records:", test.count())

# TASK 2: Text preprocessing and TF-IDF
tokenizer = RegexTokenizer(
    inputCol="message_text",
    outputCol="tokens",
    pattern="\\W+",
    minTokenLength=2
)

stop_words = StopWordsRemover(
    inputCol="tokens",
    outputCol="filtered_tokens"
)

hashing_tf = HashingTF(
    inputCol="filtered_tokens",
    outputCol="raw_features",
    numFeatures=1 << 12
)

idf = IDF(
    inputCol="raw_features",
    outputCol="features",
    minDocFreq=2
)

# TASK 3: Distributed Logistic Regression
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=30,
    regParam=0.01,
    elasticNetParam=0.0
)

pipeline = Pipeline(stages=[tokenizer, stop_words, hashing_tf, idf, lr])

print("\n========== TASK 2: TF-IDF AND TEXT PREPROCESSING ==========")
print("Tokenizer: RegexTokenizer")
print("Stop-word removal: StopWordsRemover")
print("Feature extraction: HashingTF + IDF")
print("Feature space size:", 1 << 12)

print("\n========== TASK 3: TRAINING MODEL ==========")
start = time.perf_counter()
model = pipeline.fit(train)
training_time = time.perf_counter() - start

predictions = model.transform(test).cache()
predictions.count()

print("Training time:", round(training_time, 4), "seconds")
print("Model training completed.")

# TASK 5: Evaluation metrics
accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
precision_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="precisionByLabel", metricLabel=1.0
)
recall_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="recallByLabel", metricLabel=1.0
)
f1_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)
auc_eval = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

accuracy = accuracy_eval.evaluate(predictions)
precision = precision_eval.evaluate(predictions)
recall = recall_eval.evaluate(predictions)
f1 = f1_eval.evaluate(predictions)
roc_auc = auc_eval.evaluate(predictions)

print("\n========== TASK 5: MODEL EVALUATION ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# TASK 4: Hyperparameter tuning
print("\n========== TASK 4: HYPERPARAMETER TUNING ==========")

tuning_results = []
best_score = -1.0
best_params = None

for reg in [0.001, 0.01, 0.1]:
    for iterations in [10, 30]:
        for elastic in [0.0, 0.5, 1.0]:
            candidate_lr = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=iterations,
                regParam=reg,
                elasticNetParam=elastic
            )

            candidate_pipeline = Pipeline(stages=[
                tokenizer, stop_words, hashing_tf, idf, candidate_lr
            ])

            start = time.perf_counter()
            candidate_model = candidate_pipeline.fit(train)
            elapsed = time.perf_counter() - start

            candidate_predictions = candidate_model.transform(test)

            score = f1_eval.evaluate(candidate_predictions)

            tuning_results.append(
                (reg, iterations, elastic, score, elapsed)
            )

            print(
                f"regParam={reg}, maxIter={iterations}, "
                f"elasticNetParam={elastic}, "
                f"F1={score:.4f}, time={elapsed:.2f}s"
            )

            if score > best_score:
                best_score = score
                best_params = (reg, iterations, elastic)

with open("results/hyperparameter_results.csv", "w") as f:
    f.write("regParam,maxIter,elasticNetParam,F1,training_seconds\n")
    for result in tuning_results:
        f.write(",".join(map(str, result)) + "\n")

print("Best parameters:", best_params)
print("Best tuning F1:", round(best_score, 4))

# Q5: Lower threshold to favor spam recall
print("\n========== TASK 7: CLASSIFICATION THRESHOLD ==========")

threshold_predictions = predictions.withColumn(
    "threshold_prediction",
    when(
        vector_to_array(col("probability"))[1] >= 0.30, 1.0
    ).otherwise(0.0)
)

threshold_recall_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="threshold_prediction",
    metricName="recallByLabel",
    metricLabel=1.0
)

threshold_precision_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="threshold_prediction",
    metricName="precisionByLabel",
    metricLabel=1.0
)

recall_at_030 = threshold_recall_eval.evaluate(threshold_predictions)
precision_at_030 = threshold_precision_eval.evaluate(threshold_predictions)

print("Default model threshold: 0.50")
print("Adjusted threshold: 0.30")
print(f"Spam recall at 0.30: {recall_at_030:.4f}")
print(f"Spam precision at 0.30: {precision_at_030:.4f}")

# Save batch predictions
predictions.select(
    "message_text", "label", "prediction", "probability"
).write.mode("overwrite").option("header", True).csv(
    "results/batch_predictions"
)

# Save metrics
with open("results/model_metrics.txt", "w") as f:
    f.write("SECURECOMM SMS SPAM CLASSIFICATION RESULTS\n")
    f.write("==========================================\n")
    f.write(f"Training records: {train.count()}\n")
    f.write(f"Testing records: {test.count()}\n")
    f.write(f"Training time: {training_time:.4f} seconds\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"Best parameters: {best_params}\n")
    f.write(f"Best tuning F1: {best_score:.4f}\n")
    f.write(f"Spam recall at threshold 0.30: {recall_at_030:.4f}\n")
    f.write(f"Spam precision at threshold 0.30: {precision_at_030:.4f}\n")

print("\nMetrics saved in results/model_metrics.txt")
print("Batch predictions saved in results/batch_predictions/")

predictions.unpersist()
train.unpersist()
test.unpersist()
spark.stop()

print("\n========== PRACTICAL 11 CORE TASKS COMPLETED ==========\n")
