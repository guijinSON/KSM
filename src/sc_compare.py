import pandas as pd
import json
import argparse
from sc_stats import check_correct

def process_solutions(df, no_retry_only=False):
    """Process solutions and create output list"""
    retry_n = 0
    no_retry_n = 0
    output = []
    
    for _, row in df.iterrows():
        initial_is_correct = False
        retry_is_correct = False
        no_retry_is_correct = False
        
        if not no_retry_only and 'Hmm' in row.solution:
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

def compare_solutions(df_new, df_base):
    """Compare solutions between new model (with retries) and base model"""
    retry_on_correct = 0
    retry_on_wrong = 0
    total_correct_base = 0
    total_wrong_base = 0
    
    success_after_retry_when_base_wrong = 0
    success_after_retry_when_base_correct = 0
    
    for (_, row_new), (_, row_base) in zip(df_new.iterrows(), df_base.iterrows()):
        # Check base model correctness
        base_correct = check_correct(row_base.solution, row_base.answer)
        
        if base_correct:
            total_correct_base += 1
        else:
            total_wrong_base += 1
            
        # Check if new model retries
        if 'Hmm' in row_new.solution:
            splits = row_new.solution.split('Hmm')
            initial, retry = splits[0], splits[1]
            
            if base_correct:
                retry_on_correct += 1
                if check_correct(retry, row_new.answer):
                    success_after_retry_when_base_correct += 1
            else:
                retry_on_wrong += 1
                if check_correct(retry, row_new.answer):
                    success_after_retry_when_base_wrong += 1
    
    # Calculate percentages
    retry_rate_on_correct = (retry_on_correct / total_correct_base * 100) if total_correct_base > 0 else 0
    retry_rate_on_wrong = (retry_on_wrong / total_wrong_base * 100) if total_wrong_base > 0 else 0
    
    success_rate_after_retry_base_wrong = (success_after_retry_when_base_wrong / retry_on_wrong * 100) if retry_on_wrong > 0 else 0
    success_rate_after_retry_base_correct = (success_after_retry_when_base_correct / retry_on_correct * 100) if retry_on_correct > 0 else 0
    
    return {
        'retry_patterns': {
            'retry_rate_when_base_correct': round(retry_rate_on_correct, 2),
            'retry_rate_when_base_wrong': round(retry_rate_on_wrong, 2),
            'success_rate_after_retry_base_wrong': round(success_rate_after_retry_base_wrong, 2),
            'success_rate_after_retry_base_correct': round(success_rate_after_retry_base_correct, 2)
        },
        'counts': {
            'total_correct_base': total_correct_base,
            'total_wrong_base': total_wrong_base,
            'retry_on_correct': retry_on_correct,
            'retry_on_wrong': retry_on_wrong,
            'success_after_retry_when_base_wrong': success_after_retry_when_base_wrong,
            'success_after_retry_when_base_correct': success_after_retry_when_base_correct
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Compare solution accuracy between models')
    parser.add_argument('new_model_file', type=str, help='Path to CSV file with retry capability')
    parser.add_argument('base_model_file', type=str, help='Path to CSV file without retries')
    parser.add_argument('output_file', type=str, help='Path for output JSON file')
    args = parser.parse_args()
    
    try:
        df_new = pd.read_csv(args.new_model_file)
        df_base = pd.read_csv(args.base_model_file)
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {str(e)}")
        return
    except Exception as e:
        print(f"Error reading input files: {str(e)}")
        return
    
    # Process solutions for both models
    new_output, new_retry_n, new_no_retry_n = process_solutions(df_new)
    base_output, _, _ = process_solutions(df_base, no_retry_only=True)
    
    # Calculate metrics for both models
    new_metrics = calculate_metrics(new_output)
    base_metrics = calculate_metrics(base_output)
    
    # Compare solutions
    comparison_metrics = compare_solutions(df_new, df_base)
    
    # Combine all metrics
    all_metrics = {
        'new_model': new_metrics,
        'base_model': base_metrics,
        'comparison': comparison_metrics
    }
    
    # Print results
    print("Base Model Metrics:")
    print(f"Accuracy: {base_metrics['total_accuracy']}%")
    
    print("\nNew Model Metrics:")
    print(f"Overall Accuracy: {new_metrics['total_accuracy']}%")
    print(f"Retry Accuracy: {new_metrics['retry_accuracy']}%")
    print(f"No Retry Accuracy: {new_metrics['no_retry_accuracy']}%")
    
    print("\nRetry Patterns:")
    print(f"Retry rate when base was correct: {comparison_metrics['retry_patterns']['retry_rate_when_base_correct']}%")
    print(f"Retry rate when base was wrong: {comparison_metrics['retry_patterns']['retry_rate_when_base_wrong']}%")
    print(f"Success rate after retry (when base was wrong): {comparison_metrics['retry_patterns']['success_rate_after_retry_base_wrong']}%")
    print(f"Success rate after retry (when base was correct): {comparison_metrics['retry_patterns']['success_rate_after_retry_base_correct']}%")
    
    # Save results
    try:
        with open(args.output_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(f"\nResults have been saved to '{args.output_file}'")
    except Exception as e:
        print(f"Error saving output file: {str(e)}")

if __name__ == "__main__":
    main()
