using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearningTest02
{
	class Program
	{
		private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
		private static string _dataPath => Path.Combine(AppPath, "datasets", "taxi-fare-train.csv");
		private static string _testDataPath => Path.Combine(AppPath, "datasets", "taxi-fare-test.csv");
		private static string _modelPath => Path.Combine(AppPath, "TaxiFareModel.zip");


		static async Task Main(string[] args)
		{
			Console.WriteLine("loading the model...");
			PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
			Console.WriteLine("Evaliationg...");
			Evaluate(model);
			TaxiTripFarePrediction prediction = model.Predict(TestTaxiTrips.Trip1);
			Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);
			Console.ReadLine();
		}

		public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
		{
			Console.WriteLine("1");
			var pipeline = new LearningPipeline
			{

				new TextLoader(_dataPath).CreateFrom<TaxiTrip>(separator: ','),
				new ColumnCopier(("FareAmount", "Label")),
				new CategoricalOneHotVectorizer("VendorId", "RateCode", "PaymentType"),
				new ColumnConcatenator("Features", "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType"),
				new FastTreeRegressor(),
			};
			Console.WriteLine("2");
			var model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
			Console.WriteLine("3");
			await model.WriteAsync(_modelPath);
			Console.WriteLine("4");
			return model;
		}

		public static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
		{
			var testData = new TextLoader(_testDataPath).CreateFrom<TaxiTrip>(separator: ',');
			var evaluator = new RegressionEvaluator();
			RegressionMetrics metrics = evaluator.Evaluate(model, testData);
			Console.WriteLine($"Rms = {metrics.Rms}");
			Console.WriteLine($"Rsquared = {metrics.RSquared}");
		}
	}
}
