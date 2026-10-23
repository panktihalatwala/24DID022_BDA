from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, month, to_date, desc, asc

# --------------------------------------------------
# Spark Session
# --------------------------------------------------
spark = SparkSession.builder \
    .appName("ECommerceSalesAnalysis") \
    .master("local[*]") \
    .getOrCreate()

print("\n========== PRACTICAL 6: E-COMMERCE SALES ANALYSIS ==========\n")

# --------------------------------------------------
# TASK 1: Load Sales Data
# --------------------------------------------------
df = spark.read.csv(
    "sales_data.csv",
    header=True,
    inferSchema=True
)

print("========== TASK 1: LOADED SALES DATA ==========")
df.show(15, truncate=False)

print("Total Records:", df.count())

# --------------------------------------------------
# TASK 2: Display Schema
# --------------------------------------------------
print("\n========== TASK 2: DATASET SCHEMA ==========")
df.printSchema()

# --------------------------------------------------
# TASK 3: Validate Data Types
# --------------------------------------------------
print("\n========== TASK 3: DATA TYPES ==========")
for column_name, data_type in df.dtypes:
    print(column_name, "->", data_type)

# --------------------------------------------------
# Convert Sale_Date to date
# --------------------------------------------------
df = df.withColumn(
    "Sale_Date",
    to_date(col("Sale_Date"))
)

# --------------------------------------------------
# TASK 4: Revenue = Quantity × Price
# --------------------------------------------------
df = df.withColumn(
    "Revenue",
    col("Quantity") * col("Price")
)

print("\n========== TASK 4: REVENUE CALCULATION ==========")
df.select(
    "OrderID",
    "Product",
    "Quantity",
    "Price",
    "Revenue"
).show(15)

# --------------------------------------------------
# TASK 5: Category-wise Revenue
# --------------------------------------------------
category_revenue = df.groupBy("Category") \
    .agg(sum("Revenue").alias("TotalRevenue"))

print("\n========== TASK 5: CATEGORY-WISE REVENUE ==========")
category_revenue.orderBy(
    desc("TotalRevenue")
).show()

# --------------------------------------------------
# TASK 6: Top-selling Products by Quantity
# --------------------------------------------------
top_products = df.groupBy("Product") \
    .agg(sum("Quantity").alias("QuantitySold"))

print("\n========== TASK 6: TOP-SELLING PRODUCTS ==========")
top_products.orderBy(
    desc("QuantitySold")
).show()

# --------------------------------------------------
# TASK 7: City-wise Revenue
# --------------------------------------------------
city_revenue = df.groupBy("City") \
    .agg(sum("Revenue").alias("TotalRevenue"))

print("\n========== TASK 7: CITY-WISE REVENUE ==========")
city_revenue.orderBy(
    desc("TotalRevenue")
).show()

# --------------------------------------------------
# TASK 8: Sort Categories by Revenue
# --------------------------------------------------
print("\n========== TASK 8: CATEGORIES SORTED BY REVENUE ==========")
category_revenue.orderBy(
    desc("TotalRevenue")
).show()

# --------------------------------------------------
# TASK 9: Analytical Summary
# --------------------------------------------------
print("\n========== TASK 9: ANALYTICAL SUMMARY ==========")

df.select(
    sum("Revenue").alias("TotalRevenue"),
    avg("Revenue").alias("AverageRevenuePerOrder"),
    sum("Quantity").alias("TotalQuantitySold")
).show()

print("Total Orders:", df.count())

# --------------------------------------------------
# TASK 10: Business Insights
# --------------------------------------------------
print("\n========== TASK 10: BUSINESS INSIGHTS ==========")

highest_category = category_revenue.orderBy(
    desc("TotalRevenue")
).first()

top_product = top_products.orderBy(
    desc("QuantitySold")
).first()

highest_city = city_revenue.orderBy(
    desc("TotalRevenue")
).first()

least_category = category_revenue.orderBy(
    asc("TotalRevenue")
).first()

print("Highest Revenue Category:", highest_category["Category"])
print("Highest Category Revenue:", highest_category["TotalRevenue"])

print("Top-Selling Product:", top_product["Product"])
print("Top-Selling Quantity:", top_product["QuantitySold"])

print("Highest Revenue City:", highest_city["City"])
print("Highest City Revenue:", highest_city["TotalRevenue"])

print("Least-Performing Category:", least_category["Category"])
print("Least Category Revenue:", least_category["TotalRevenue"])

# --------------------------------------------------
# SUPPLEMENTARY 1: Top 3 Revenue-Generating Cities
# --------------------------------------------------
print("\n========== SUPPLEMENTARY 1: TOP 3 CITIES ==========")
city_revenue.orderBy(
    desc("TotalRevenue")
).show(3)

# --------------------------------------------------
# SUPPLEMENTARY 2: Average Revenue per Order
# --------------------------------------------------
print("\n========== SUPPLEMENTARY 2: AVERAGE REVENUE PER ORDER ==========")
df.select(
    avg("Revenue").alias("AverageRevenuePerOrder")
).show()

# --------------------------------------------------
# SUPPLEMENTARY 3: Discounted Revenue
# --------------------------------------------------
df_discount = df.withColumn(
    "DiscountAmount",
    col("Revenue") * 0.10
).withColumn(
    "FinalRevenue",
    col("Revenue") - col("DiscountAmount")
)

print("\n========== SUPPLEMENTARY 3: DISCOUNTED REVENUE ==========")
df_discount.select(
    "OrderID",
    "Revenue",
    "DiscountAmount",
    "FinalRevenue"
).show(15)

# --------------------------------------------------
# SUPPLEMENTARY 4: Least-performing Category
# --------------------------------------------------
print("\n========== SUPPLEMENTARY 4: LEAST-PERFORMING CATEGORY ==========")
category_revenue.orderBy(
    asc("TotalRevenue")
).show(1)

# --------------------------------------------------
# SUPPLEMENTARY 5: Monthly Revenue Summary
# --------------------------------------------------
monthly_revenue = df.withColumn(
    "Month",
    month("Sale_Date")
).groupBy("Month") \
 .agg(sum("Revenue").alias("MonthlyRevenue")) \
 .orderBy("Month")

print("\n========== SUPPLEMENTARY 5: MONTHLY REVENUE ==========")
monthly_revenue.show()

# --------------------------------------------------
# Save important outputs
# --------------------------------------------------
category_revenue.orderBy(desc("TotalRevenue")) \
    .coalesce(1) \
    .write.mode("overwrite").option("header", True) \
    .csv("output_category_revenue")

top_products.orderBy(desc("QuantitySold")) \
    .coalesce(1) \
    .write.mode("overwrite").option("header", True) \
    .csv("output_top_products")

city_revenue.orderBy(desc("TotalRevenue")) \
    .coalesce(1) \
    .write.mode("overwrite").option("header", True) \
    .csv("output_city_revenue")

monthly_revenue.coalesce(1) \
    .write.mode("overwrite").option("header", True) \
    .csv("output_monthly_revenue")

print("\n========== PRACTICAL 6 COMPLETED ==========\n")

spark.stop()
