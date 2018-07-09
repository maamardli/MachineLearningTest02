using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearningTest02
{
	class Program
	{
		static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
		static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
		static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

		static async Task Main(string[] args)
		{
			PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
			Evaluate(model);
		}

		public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
		{
			var pipeline = new LearningPipeline
			{
				new TextLoader(_dataPath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','),
				new ColumnCopier("FareAmount", "Label"),
				new CategoricalOneHotVectorizer("VendorId", "RateCocde", "PaymentType"),
				new ColumnConcatenator("Feature", "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType"),
				new FastTreeRegressor(),
			};
			PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
			await model.WriteAsync(_modelPath);
			return model;
		}

		public static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
		{
			var testData = new TextLoader(_testDataPath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
			var evaluator = new RegressionEvaluator();
			RegressionMetrics metrics = evaluator.Evaluate(model, testData);
			Console.WriteLine($"Rms = {metrics.Rms}");
			Console.WriteLine($"Rsquared = {metrics.RSquared}");
		}
	}
}
