using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

public class Program
{
    public static void Main()
    {
        // Initialize MLContext
        MLContext mlContext = new MLContext();

        // Generate random data 
        var data = GenerateRandomData(mlContext, 100); // Generate 100 samples

        // Define pipeline and train model
        var model = TrainModel(mlContext, data);

        // Evaluate the model
        EvaluateModel(mlContext, model, data);

        // Test Predictions 
        //TestPredictions(model);
    }

    // generate random synthetic data
    static IDataView GenerateRandomData(MLContext mlContext, int sampleCount)
    {
        var random = new Random();
        var data = new List<NetworkTrafficData>();

        for (int i = 0; i < sampleCount; i++)
        {
            bool isAnomaly = random.NextDouble() < 0.2; 

            data.Add(new NetworkTrafficData
            {
                PacketCount = isAnomaly ? random.Next(500, 1000) : random.Next(80, 300),
                AveragePacketSize = isAnomaly ? random.Next(1500, 3000) : random.Next(400, 800),
                PacketDuration = isAnomaly ? random.Next(200, 400) : random.Next(50, 150),
                IntervalBetweenPackets = isAnomaly ? random.Next(10, 30) : random.Next(40, 80),
                PacketFrequency = isAnomaly ? random.Next(7, 15) : random.Next(3, 6),
                TotalDataSent = isAnomaly ? random.Next(7000, 14000) : random.Next(2000, 5000),
                SourceDestinationRatio = isAnomaly ? random.Next(2, 5) : random.Next(1, 2),
                Label = !isAnomaly // True for normal, false for anomaly
            });
        }

        return mlContext.Data.LoadFromEnumerable(data);
    }

    // train model using generated data
    static ITransformer TrainModel(MLContext mlContext, IDataView data)
    {
        var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
            nameof(NetworkTrafficData.PacketCount),
            nameof(NetworkTrafficData.AveragePacketSize),
            nameof(NetworkTrafficData.PacketDuration),
            nameof(NetworkTrafficData.IntervalBetweenPackets),
            nameof(NetworkTrafficData.PacketFrequency),
            nameof(NetworkTrafficData.TotalDataSent),
            nameof(NetworkTrafficData.SourceDestinationRatio))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"));

        var trainer = mlContext.BinaryClassification.Trainers.FastTree(
            labelColumnName: "Label",
            featureColumnName: "Features",
            numberOfLeaves: 20,
            learningRate: 0.1,
            numberOfTrees: 100);

        var trainingPipeline = dataProcessPipeline.Append(trainer);
        return trainingPipeline.Fit(data);
    }

    // evaluate model accuracy
    static void EvaluateModel(MLContext mlContext, ITransformer model, IDataView data)
    {
        var predictions = model.Transform(data);
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

        Console.WriteLine($"Model accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
    }

    // Method to test predictions on new data samples
    static void TestPredictions(ITransformer model)
    {
        var mlContext = new MLContext();
        var predictionEngine = mlContext.Model.CreatePredictionEngine<NetworkTrafficData, AnomalyPrediction>(model);

        var testData = new[]
        {
            new NetworkTrafficData { PacketCount = 150, AveragePacketSize = 500 },  // Normal traffic
            new NetworkTrafficData { PacketCount = 400, AveragePacketSize = 1500 }  // Anomaly
        };

        Console.WriteLine("Predictions:");
        foreach (var traffic in testData)
        {
            var prediction = predictionEngine.Predict(traffic);
            string result = prediction.Prediction ? "Normal" : "Anomaly";
            Console.WriteLine($"PacketCount: {traffic.PacketCount}, AveragePacketSize: {traffic.AveragePacketSize}");
            Console.WriteLine($"Prediction: {result}, Score: {prediction.Score}, Probability: {prediction.Probability:P2}");
        }
    }
}

