---

### **Step 1: Create a Directory for Advanced Analysis**
First, create a directory to store all your advanced analysis scripts.

```bash
mkdir advanced_analysis
cd advanced_analysis
```

---

### **Step 2: Handle Imbalanced Data**
This script will balance the data by oversampling the minority class.

#### **1. Create the script**
```bash
nano handle_imbalanced_data.py
```

#### **2. Add this code:**
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Initialize Spark session
spark = SparkSession.builder \
    .appName("HandleImbalancedData") \
    .getOrCreate()

# Load cleaned data
data_path = "../data/credit_card_transactions.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Class distribution
class_distribution = df.groupBy("FraudLabel").count()
class_distribution.show()

# Oversample minority class
fraudulent = df.filter(col("FraudLabel") == 1)
non_fraudulent = df.filter(col("FraudLabel") == 0)
oversampled_fraud = fraudulent.sample(withReplacement=True, fraction=3.0)
balanced_data = non_fraudulent.union(oversampled_fraud)

# Verify new class distribution
balanced_data.groupBy("FraudLabel").count().show()

# Save balanced data for further analysis
balanced_data.write.csv("../balanced_data", header=True)
```

#### **3. Run the script:**
```bash
spark-submit handle_imbalanced_data.py
```

---

### **Step 3: Analyze Feature Importance**
This script will determine which features contribute the most to fraud detection.

#### **1. Create the script**
```bash
nano feature_importance.py
```

#### **2. Add this code:**
```python
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FeatureImportance") \
    .getOrCreate()

# Load balanced data
data_path = "../balanced_data"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Train a Random Forest model
rf = RandomForestClassifier(labelCol="FraudLabel", featuresCol="features", numTrees=100)
rf_model = rf.fit(df)

# Extract feature importance
feature_importances = rf_model.featureImportances.toArray()
feature_names = ["TransactionAmount", "HighValueTransaction", "TransactionHour", 
                 "UserAge", "TransactionTypeIndex", "CardTypeIndex"]

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Fraud Detection")
plt.savefig("feature_importance.png")
```

#### **3. Run the script:**
```bash
spark-submit feature_importance.py
```
Check the generated plot in `feature_importance.png`.

---

### **Step 4: Analyze Fraud Trends**
This script will explore fraud trends over time, transaction types, and other dimensions.

#### **1. Create the script**
```bash
nano fraud_trends.py
```

#### **2. Add this code:**
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import dayofweek, month

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FraudTrends") \
    .getOrCreate()

# Load cleaned data
data_path = "../data/credit_card_transactions.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Fraud by day of the week
fraud_by_day = df.filter(col("FraudLabel") == 1) \
    .groupBy(dayofweek("TransactionDate").alias("DayOfWeek")).count()
fraud_by_day.show()

# Fraud by transaction type
fraud_by_type = df.filter(col("FraudLabel") == 1) \
    .groupBy("TransactionType").count()
fraud_by_type.show()

# Fraud by month
fraud_by_month = df.filter(col("FraudLabel") == 1) \
    .groupBy(month("TransactionDate").alias("Month")).count()
fraud_by_month.show()
```

#### **3. Run the script:**
```bash
spark-submit fraud_trends.py
```

---

### **Step 5: Simulate Real-Time Fraud Detection**
This script will simulate a streaming pipeline for fraud detection.

#### **1. Create the script**
```bash
nano real_time_fraud_detection.py
```

#### **2. Add this code:**
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, hour

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RealTimeFraudDetection") \
    .getOrCreate()

# Define schema
schema = "TransactionID STRING, UserID STRING, TransactionAmount DOUBLE, TransactionDate TIMESTAMP, \
          Merchant STRING, TransactionType STRING, UserAge INT, UserCountry STRING, CardType STRING"

# Read streaming data
streaming_data = spark.readStream.schema(schema).csv("../streaming_data")

# Preprocess streaming data
streaming_data = streaming_data.withColumn(
    "HighValueTransaction", when(col("TransactionAmount") > 1000, 1).otherwise(0)
).withColumn("TransactionHour", hour(col("TransactionDate")))

# Load trained model and predict
from pyspark.ml.classification import RandomForestClassificationModel
rf_model = RandomForestClassificationModel.load("../rf_model")
predictions = rf_model.transform(streaming_data)

# Output results to console
query = predictions.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
```

#### **3. Run the script:**
```bash
spark-submit real_time_fraud_detection.py
```

---

### **Step 6: Perform Anomaly Detection**
This script will identify unusual transactions using clustering.

#### **1. Create the script**
```bash
nano anomaly_detection.py
```

#### **2. Add this code:**
```python
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans

# Initialize Spark session
spark = SparkSession.builder \
    .appName("AnomalyDetection") \
    .getOrCreate()

# Load data
data_path = "../data/credit_card_transactions.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Prepare data for clustering
anomaly_data = df.select("TransactionAmount", "UserAge", "TransactionHour").na.drop()

# Apply K-Means clustering
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(anomaly_data)

# Make predictions
clustered_data = model.transform(anomaly_data)
clustered_data.show()
```

#### **3. Run the script:**
```bash
spark-submit anomaly_detection.py
```

---

