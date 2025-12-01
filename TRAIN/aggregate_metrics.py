import os
import argparse
import pandas as pd

def aggregate_metrics(input_dir, output_file):
    print(f"Scanning directory: {input_dir} for 'metrics.txt' files...")
    
    aggregated_data = []

    for root, _, files in os.walk(input_dir):
        if "metrics.txt" in files:
            file_path = os.path.join(root, "metrics.txt")      
            model_name = os.path.basename(root)
            model_metrics = {'Model': model_name}
            
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if ":" in line:
                            key, value = line.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            
                            if key in ['BLEU', 'STA', 'SIM', 'FL', 'J']:
                                try:
                                    model_metrics[key] = round(float(value), 4)
                                except ValueError:
                                    pass
                
                if len(model_metrics) > 1:
                    aggregated_data.append(model_metrics)
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if not aggregated_data:
        print("No metrics.txt files found!")
        return

    df = pd.DataFrame(aggregated_data)

    desired_order = ['Model', 'BLEU', 'STA', 'SIM', 'FL', 'J']
    cols = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in cols]
    df = df[cols + remaining]

    if 'J' in df.columns:
        df = df.sort_values(by='J', ascending=False)

    df.to_csv(output_file, index=False)
    
    print(f"Aggregated metrics for {len(df)} models.")
    print(f"Summary saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./evaluation_outputs", 
                        help="Root directory containing model output folders")
    parser.add_argument("--output_file", type=str, default="summary_metrics.csv", 
                        help="Name of the output CSV file")
    
    args = parser.parse_args()
    
    aggregate_metrics(args.input_dir, args.output_file)