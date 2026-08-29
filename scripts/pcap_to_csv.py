# pcap_to_csv.py
# Converts pcap to flow CSV — alternative to CICFlowMeter
from nfstream import NFStreamer
import pandas as pd

print("Reading pcap and extracting flows...")
streamer = NFStreamer(
    source="Wednesday-workingHours.pcap",
    statistical_analysis=True
)

df = streamer.to_pandas()
print(f"Extracted {len(df):,} flows")
print(f"Columns: {list(df.columns)}")

df.to_csv("cicflow_output/Wednesday-workingHours.pcap_ISCX.csv", index=False)
print("Saved to cicflow_output/Wednesday-workingHours.pcap_ISCX.csv")
