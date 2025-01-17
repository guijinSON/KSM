import pandas as pd
import json
import argparse
from data import answer_in_last_sentence, parse_boxed_value

def check_correct(solution, answer):
    if any([answer_in_last_sentence(solution,answer),parse_boxed_value(solution,answer)]):
        return True
    else:
        return False

def process_solutions(df):
    """Process solutions and create output list"""
    retry_n = 0
    no_retry_n = 0
    output = []
    
    for _, row in df.iterrows():
        initial_is_correct = False
        retry_is_correct = False
        no_retry_is_correct = False
        
        if 'Hmm' in row.solution:
            splits = row.solution.split('Hmm')
            initial, retry = splits[0], splits[1]
            retry_n += 1
            if check_correct(initial, row.answer):
                initial_is_correct = True
            if check_correct(retry, row.answer):
                retry_is_correct = True
            no_retry_is_correct = None
        else:
            no_retry = row.solution
            if check_correct(no_retry, row.answer):
                no_retry_is_correct = True
            initial_is_correct = None
            retry_is_correct = None
            no_retry_n += 1
            
        output.append([initial_is_correct, retry_is_correct, no_retry_is_correct])
    
    return output, retry_n, no_retry_n

def calculate_metrics(output):
    """Calculate accuracy metrics"""
    total_samples = len(output)
    total_correct = 0
    retry_correct = 0
    retry_wrong = 0
    retry_total = 0
    no_retry_correct = 0
    no_retry_total = 0
    
    for initial, retry, no_retry in output:
        # Case 1: Solution with retry
        if retry is not None:  # This means it was a retry case
            retry_total += 1
            if retry:  # If retry was correct
                total_correct += 1
                retry_correct += 1
            else:
                retry_wrong += 1
                
        # Case 2: Solution without retry
        else:  # This means it was a no_retry case
            no_retry_total += 1
            if no_retry:  # If the single attempt was correct
                total_correct += 1
                no_retry_correct += 1
    
    # Calculate metrics
    total_accuracy = (total_correct / total_samples) * 100
    retry_accuracy = (retry_correct / retry_total) * 100 if retry_total > 0 else 0
    no_retry_accuracy = (no_retry_correct / no_retry_total) * 100 if no_retry_total > 0 else 0
    
    # Calculate retry success rate
    retry_success_rate = (retry_correct / retry_total) * 100 if retry_total > 0 else 0
    retry_failure_rate = (retry_wrong / retry_total) * 100 if retry_total > 0 else 0
    
    return {
        'total_accuracy': round(total_accuracy, 2),
        'retry_accuracy': round(retry_accuracy, 2),
        'no_retry_accuracy': round(no_retry_accuracy, 2),
        'retry_success_rate': round(retry_success_rate, 2),
        'retry_failure_rate': round(retry_failure_rate, 2),
        'counts': {
            'total_samples': total_samples,
            'retry_total': retry_total,
            'no_retry_total': no_retry_total,
            'retry_correct': retry_correct,
            'retry_wrong': retry_wrong
        }
    }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze solution accuracy with retries')
    parser.add_argument('input_file', type=str, help='Path to input CSV file')
    parser.add_argument('output_file', type=str, help='Path for output JSON file')
    args = parser.parse_args()
    
    # Read DataFrame
    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found")
        return
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return
    
    # Process solutions
    output, retry_n, no_retry_n = process_solutions(df)
    
    # Calculate metrics
    metrics = calculate_metrics(output)
    
    # Print results
    print(f"Overall Accuracy: {metrics['total_accuracy']}%")
    print(f"\nBreakdown by retry status:")
    print(f"Retry Accuracy: {metrics['retry_accuracy']}%")
    print(f"No Retry Accuracy: {metrics['no_retry_accuracy']}%")
    print(f"\nRetry Performance:")
    print(f"Success Rate when Retrying: {metrics['retry_success_rate']}%")
    print(f"Failure Rate when Retrying: {metrics['retry_failure_rate']}%")
    print(f"\nCounts:")
    print(f"Total samples: {metrics['counts']['total_samples']}")
    print(f"Total retries: {metrics['counts']['retry_total']}")
    print(f"Successful retries: {metrics['counts']['retry_correct']}")
    print(f"Failed retries: {metrics['counts']['retry_wrong']}")
    
    # Save results to JSON
    try:
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\nResults have been saved to '{args.output_file}'")
    except Exception as e:
        print(f"Error saving output file: {str(e)}")

if __name__ == "__main__":
    main()
