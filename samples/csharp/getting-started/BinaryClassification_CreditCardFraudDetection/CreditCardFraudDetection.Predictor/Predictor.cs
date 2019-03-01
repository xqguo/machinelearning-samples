using CreditCardFraudDetection.Common;
using CreditCardFraudDetection.Common.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

namespace CreditCardFraudDetection.Predictor
{
    public class Predictor
    {
        private readonly string _modelfile;
        private readonly string _dasetFile;

        public Predictor(string modelfile, string dasetFile) {
            _modelfile = modelfile ?? throw new ArgumentNullException(nameof(modelfile));
            _dasetFile = dasetFile ?? throw new ArgumentNullException(nameof(dasetFile));
        }

        public void RunMultiplePredictions(int numberOfTransactions, int? seed = 1) {

            var mlContext = new MLContext(seed);

            TextLoader.Column[] columnsPlus = new[] {
                    // A boolean column depicting the 'label'.
                    new TextLoader.Column("Label", DataKind.Boolean, 0),
                    // 30 Features V1..V28 + Amount + StratificationColumn
                    new TextLoader.Column("V1", DataKind.Single, 1 ),
                    new TextLoader.Column("V2", DataKind.Single, 2 ),
                    new TextLoader.Column("V3", DataKind.Single, 3 ),
                    new TextLoader.Column("V4", DataKind.Single, 4 ),
                    new TextLoader.Column("V5", DataKind.Single, 5 ),
                    new TextLoader.Column("V6", DataKind.Single, 6 ),
                    new TextLoader.Column("V7", DataKind.Single, 7 ),
                    new TextLoader.Column("V8", DataKind.Single, 8 ),
                    new TextLoader.Column("V9", DataKind.Single, 9 ),
                    new TextLoader.Column("V10", DataKind.Single, 10 ),
                    new TextLoader.Column("V11", DataKind.Single, 11 ),
                    new TextLoader.Column("V12", DataKind.Single, 12 ),
                    new TextLoader.Column("V13", DataKind.Single, 13 ),
                    new TextLoader.Column("V14", DataKind.Single, 14 ),
                    new TextLoader.Column("V15", DataKind.Single, 15 ),
                    new TextLoader.Column("V16", DataKind.Single, 16 ),
                    new TextLoader.Column("V17", DataKind.Single, 17 ),
                    new TextLoader.Column("V18", DataKind.Single, 18 ),
                    new TextLoader.Column("V19", DataKind.Single, 19 ),
                    new TextLoader.Column("V20", DataKind.Single, 20 ),
                    new TextLoader.Column("V21", DataKind.Single, 21 ),
                    new TextLoader.Column("V22", DataKind.Single, 22 ),
                    new TextLoader.Column("V23", DataKind.Single, 23 ),
                    new TextLoader.Column("V24", DataKind.Single, 24 ),
                    new TextLoader.Column("V25", DataKind.Single, 25 ),
                    new TextLoader.Column("V26", DataKind.Single, 26 ),
                    new TextLoader.Column("V27", DataKind.Single, 27 ),
                    new TextLoader.Column("V28", DataKind.Single, 28 ),
                    new TextLoader.Column("Amount", DataKind.Single, 29 ),
                    new TextLoader.Column("StratificationColumn", DataKind.Single, 30 )
                };

            //LoaderOptimization test data into DataView
            var dataTest = mlContext.Data.LoadFromTextFile(_dasetFile,
                                                           columnsPlus,
                                                           hasHeader: true,
                                                           separatorChar: ',');

            //Inspect/Peek data from datasource
            ConsoleHelpers.ConsoleWriterSection($"Inspect {numberOfTransactions} transactions observed as fraud and {numberOfTransactions} not observed as fraud, from the test datasource:");
            ConsoleHelpers.InspectData(mlContext, dataTest, numberOfTransactions);

            ConsoleHelpers.ConsoleWriteHeader($"Predictions from saved model:");

            ITransformer model;
            using (var file = File.OpenRead(_modelfile))
            {
                model = mlContext.Model.Load(file);
            }

            var predictionEngine = model.CreatePredictionEngine<TransactionObservation, TransactionFraudPrediction>(mlContext);
            ConsoleHelpers.ConsoleWriterSection($"Test {numberOfTransactions} transactions, from the test datasource, that should be predicted as fraud (true):");
            mlContext.Data.CreateEnumerable<TransactionObservation>(dataTest, reuseRowObject: false)
                        .Where(x => x.Label == true)
                        .Take(numberOfTransactions)
                        .Select(testData => testData)
                        .ToList()
                        .ForEach(testData => 
                                    {
                                        Console.WriteLine($"--- Transaction ---");
                                        testData.PrintToConsole();
                                        predictionEngine.Predict(testData).PrintToConsole();
                                        Console.WriteLine($"-------------------");
                                    });


             ConsoleHelpers.ConsoleWriterSection($"Test {numberOfTransactions} transactions, from the test datasource, that should NOT be predicted as fraud (false):");
             mlContext.Data.CreateEnumerable<TransactionObservation>(dataTest, reuseRowObject: false)
                        .Where(x => x.Label == false)
                        .Take(numberOfTransactions)
                        .ToList()
                        .ForEach(testData =>
                                    {
                                        Console.WriteLine($"--- Transaction ---");
                                        testData.PrintToConsole();
                                        predictionEngine.Predict(testData).PrintToConsole();
                                        Console.WriteLine($"-------------------");
                                    });
        }
     
    }
}
