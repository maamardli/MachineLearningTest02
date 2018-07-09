﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningTest02
{
	static class TestTaxiTrips
	{
		internal static readonly TaxiTrip Trip1 = new TaxiTrip
		{
			VendorId = "VTS",
			RateCode = "1",
			PassengerCount = 1,
			TripDistance = 10.33f,
			PaymentType = "CSH",
			FareAmount = 0 // predict it. actual = 29.5
		};
	}
}
