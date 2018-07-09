using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Api;

namespace MachineLearningTest02
{
	class TaxiTripFarePrediction
	{
		[ColumnName("Score")]
		public float FareAmount;
	}
}
