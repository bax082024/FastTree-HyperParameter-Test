using Microsoft.ML.Data;

public class AnomalyPrediction
{
  [ColumnName("PredictedLabel")]
  
    public bool Prediction { get; set; }
    public float Score { get; set; }
    public float Probability { get; set; }
  
}