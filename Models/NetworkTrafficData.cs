using Microsoft.ML.Data;

public class NetworkTrafficData
{
  [LoadColumn(0)]
  public float PacketCount { get; set; }

  [LoadColumn(1)]
  public float AveragePacketSize { get; set; }

  [LoadColumn(2)]
  public float PacketDuration { get; set; } // true for normal traffic and false for anomaly

  [LoadColumn(3)]
  public float IntervalBetweenPackets { get; set; }

  [LoadColumn(4)]
  public float PacketFrequency { get; set; }

  [LoadColumn(5)]
  public float TotalDataSent { get; set; }

  [LoadColumn(6)]
  public float SourceDestinationRatio { get; set; }

  [LoadColumn(7)]
  public bool Label { get; set; }
}