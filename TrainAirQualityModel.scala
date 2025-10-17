import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import breeze.plot._
import breeze.linalg._
import scala.sys.process._

object TrainAirQualityModel {

  def main(args: Array[String]): Unit = {

    // 1️⃣ Create Spark session
    val spark = SparkSession.builder()
      .appName("AirQualityModel_Training")
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    // 2️⃣ Load cleaned dataset from HDFS
    val inputPath = "hdfs://localhost:9000/BDA1/Preprocessed_Dataset/china-AQ-dataset.csv"
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputPath)

    println("✅ Dataset Loaded Successfully")
    df.show(5)

    // 3️⃣ Define pollutants and create next-hour targets
    val pollutants = Seq("pm10", "pm25", "o3", "co", "no2", "so2")
    val windowSpec = Window.partitionBy("location_id").orderBy("datetime")

    val dfTargets = pollutants.foldLeft(df) { (tempDF, colName) =>
      tempDF.withColumn(s"${colName}_next", lead(col(colName), 1).over(windowSpec))
    }.na.drop()

    // 4️⃣ Feature columns (based on user-specified inputs)
    val featureCols = Array("pm10", "pm25", "o3", "co", "no2", "so2", "hour", "location_id")

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    // 5️⃣ Evaluation metrics containers
    var pollutantNames = List[String]()
    var r2Scores = List[Double]()
    var maeScores = List[Double]()

    val evaluatorR2 = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("r2")

    val evaluatorMAE = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("mae")

    // 6️⃣ Train Random Forest model for each pollutant
    pollutants.foreach { pol =>
      println(s"\n🔹 Training model for $pol prediction...")

      val dfPrepared = assembler.transform(dfTargets)
        .select("features", s"${pol}_next")
        .withColumnRenamed(s"${pol}_next", "label")

      val Array(train, test) = dfPrepared.randomSplit(Array(0.8, 0.2), seed = 42)

      val rf = new RandomForestRegressor()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setNumTrees(200)
        .setMaxDepth(10)
        .setFeatureSubsetStrategy("sqrt")
        .setSubsamplingRate(0.8)

      val model = rf.fit(train)
      val predictions = model.transform(test)

      val r2 = evaluatorR2.evaluate(predictions)
      val mae = evaluatorMAE.evaluate(predictions)

      println(f"✅ $pol%-5s | R² = $r2%1.4f | MAE = $mae%1.4f")

      // Collect metrics
      pollutantNames = pollutantNames :+ pol
      r2Scores = r2Scores :+ r2
      maeScores = maeScores :+ mae

      // Save trained model to HDFS
      val outputPath = s"hdfs://localhost:9000/BDA1/Models/${pol}_model"
      model.write.overwrite().save(outputPath)
      println(s"💾 Saved model for $pol → $outputPath")
    }

    // 7️⃣ Save metrics as CSV
    import spark.implicits._
    val metricsDF = pollutantNames.zip(r2Scores.zip(maeScores))
      .map { case (pol, (r2, mae)) => (pol, r2, mae) }
      .toDF("pollutant", "r2", "mae")

    val metricsPath = "hdfs://localhost:9000/BDA1/Metrics/model_metrics.csv"
    metricsDF.coalesce(1)
      .write.mode("overwrite").option("header", "true").csv(metricsPath)

    println(s"\n📊 Metrics saved → $metricsPath")
    println("\n✅ All pollutant models trained, evaluated, and saved successfully!")
    
    try {
    println("📊 Generating visualization plots...")

      // Collect metrics into local driver
      val r2 = DenseVector(metricsDF.select("r2").as[Double].collect())
      val mae = DenseVector(metricsDF.select("mae").as[Double].collect())
      val x = DenseVector((1 to pollutants.length).map(_.toDouble).toArray)

      // ---------- 1️⃣ Model Performance (Bar / Scatter)
      val f1 = Figure("Model Performance per Pollutant")
      val p1 = f1.subplot(0)
      p1 += plot(x, r2, name = "R²", colorcode = "blue")
      p1 += plot(x, mae, name = "MAE", colorcode = "red")
      p1.legend = true
      p1.xlabel = "Pollutant"
      p1.ylabel = "Score"
      p1.title = "Model Performance per Pollutant"
      f1.saveas("model_performance.png")
    } catch {
      case e: Exception => println(s"⚠️ Plot generation failed: ${e.getMessage}")
    }
  }
}
