---

### **Step 1: Create the Project Directory Structure**

1. **Navigate to Your Workspace**:
   - Run the following command to move to your workspace:
     ```bash
     cd ~/pyspark_social_media_project
     ```

2. **Create Project Directories**:
   - Organize your project with the following structure:
     ```bash
     mkdir -p credit_card_fraud_detection/{data,scripts,logs,output}
     ```
     - `data`: Store input and processed data files.
     - `scripts`: Store Python/PySpark scripts.
     - `logs`: Store application logs.
     - `output`: Store generated output files like predictions.

3. **Verify the Structure**:
   - Use the `tree` command to confirm:
     ```bash
     tree credit_card_fraud_detection
     ```

   Output should look like:
   ```
   credit_card_fraud_detection
   ├── data
   ├── logs
   ├── output
   └── scripts
   ```

---

### **Step 2: Generate Synthetic Data**

1. **Create a Data Generation Script**:
   - Navigate to the `scripts` directory:
     ```bash
     cd credit_card_fraud_detection/scripts
     ```

   - Create the file:
     ```bash
     nano generate_data.py
     ```

   - Paste the following code into `generate_data.py`:
     ```python
     import csv
     from faker import Faker
     import random

     def generate_credit_card_data(filename, rows):
         fake = Faker()
         fields = [
             'TransactionID', 'UserID', 'TransactionAmount', 
             'TransactionDate', 'Merchant', 'TransactionType', 
             'FraudLabel', 'UserAge', 'UserCountry', 'CardType'
         ]
         transaction_types = ['POS', 'E-commerce', 'ATM', 'Mobile']
         card_types = ['Credit', 'Debit', 'Prepaid']

         with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
             csvwriter = csv.writer(csvfile)
             csvwriter.writerow(fields)

             for _ in range(rows):
                 transaction_id = fake.uuid4()
                 user_id = fake.uuid4()
                 amount = round(random.uniform(5, 2000), 2)
                 transaction_date = fake.date_time_this_year().isoformat()
                 merchant = fake.company()
                 transaction_type = random.choice(transaction_types)
                 fraud_label = random.choice([0, 1])  # 0 for genuine, 1 for fraud
                 user_age = random.randint(18, 75)
                 user_country = fake.country()
                 card_type = random.choice(card_types)

                 csvwriter.writerow([
                     transaction_id, user_id, amount, transaction_date, 
                     merchant, transaction_type, fraud_label, user_age, 
                     user_country, card_type
                 ])

     # Generate synthetic data
     if __name__ == "__main__":
         generate_credit_card_data('../data/credit_card_transactions.csv', 1000000)
         print("Synthetic data generation completed!")
     ```

   - Save and close (`Ctrl + O`, then `Ctrl + X`).

2. **Run the Script**:
   - Navigate to the project root directory:
     ```bash
     cd ~/pyspark_social_media_project/credit_card_fraud_detection
     ```

   - Execute the script to generate data:
     ```bash
     python3 scripts/generate_data.py
     ```

   - Confirm that the file `credit_card_transactions.csv` has been generated in the `data` directory:
     ```bash
     ls data
     ```

---

### **Step 3: Set Up the PySpark Job**

1. **Create the PySpark Job Script**:
   - Navigate to the `scripts` folder:
     ```bash
     cd scripts
     ```

   - Create a new file named `fraud_detection.py`:
     ```bash
     nano fraud_detection.py
     ```

2. **Paste the Following Code**:
   ```python
   from pyspark.sql import SparkSession
   from pyspark.sql.functions import col, to_date, when, hour
   from pyspark.ml.feature import StringIndexer, VectorAssembler
   from pyspark.ml.classification import RandomForestClassifier
   from pyspark.ml.evaluation import MulticlassClassificationEvaluator

   # Initialize Spark Session
    spark = SparkSession.builder \
    .appName("CreditCardFraudDetection") \
    .master("local[*]") \
    .getOrCreate()


   # Load data
   data_path = "file:///home/alam/pyspark_social_media_project/credit_card_fraud_detection/data/credit_card_transactions.csv"
   df = spark.read.csv(data_path, header=True, inferSchema=True)

   # Data Cleaning
   df_cleaned = df.dropna().dropDuplicates()
   df_cleaned = df_cleaned.withColumn("TransactionDate", to_date(col("TransactionDate")))

   # Feature Engineering
   df_cleaned = df_cleaned.withColumn(
       "HighValueTransaction", when(col("TransactionAmount") > 1000, 1).otherwise(0)
   )
   df_cleaned = df_cleaned.withColumn("TransactionHour", hour(col("TransactionDate")))

   # Encoding categorical variables
   indexer = StringIndexer(
       inputCols=["TransactionType", "CardType"], 
       outputCols=["TransactionTypeIndex", "CardTypeIndex"]
   )
   df_encoded = indexer.fit(df_cleaned).transform(df_cleaned)

   # Assemble features
   assembler = VectorAssembler(
       inputCols=["TransactionAmount", "HighValueTransaction", "TransactionHour", "UserAge", "TransactionTypeIndex", "CardTypeIndex"],
       outputCol="features"
   )
   df_prepared = assembler.transform(df_encoded)

   # Select data for training
   df_final = df_prepared.select("features", "FraudLabel")

   # Split data into train and test sets
   train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

   # Train a Random Forest Classifier
   rf = RandomForestClassifier(labelCol="FraudLabel", featuresCol="features", numTrees=100)
   rf_model = rf.fit(train_data)

   # Test the model
   predictions = rf_model.transform(test_data)

   # Evaluate the model
   evaluator = MulticlassClassificationEvaluator(
       labelCol="FraudLabel", predictionCol="prediction", metricName="accuracy"
   )
   accuracy = evaluator.evaluate(predictions)
   print(f"Model Accuracy: {accuracy}")

   # Save predictions
   predictions.select("FraudLabel", "prediction", "probability").write.csv("../output/predictions.csv", header=True, mode="overwrite")
   print("Predictions saved!")
   ```

   - Save and close (`Ctrl + O`, then `Ctrl + X`).

3. **Run the PySpark Job**:
   - Go to the project root directory:
     ```bash
     cd ~/pyspark_social_media_project/credit_card_fraud_detection
     ```

   - Run the PySpark job:
     ```bash
     spark-submit scripts/fraud_detection.py
     ```

---

### **Step 4: Verify Outputs**

1. **Check Logs**:
   - Logs will appear in the terminal. If needed, redirect them to a file in the `logs` folder:
     ```bash
     spark-submit scripts/fraud_detection.py > logs/job.log 2>&1
     ```

2. **Review Predictions**:
   - Check the `output` directory for `predictions.csv`:
     ```bash
     ls output
     ```

3. **Inspect Data**:
   - Open the `predictions.csv` file to verify fraud detection results:
     ```bash
     less output/predictions.csv
     ```

---
