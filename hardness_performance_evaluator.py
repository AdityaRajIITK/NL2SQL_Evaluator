import json
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from statistics import mean
import re

from sql_hardness_classifier import SQLHardnessClassifier
from partial_component_matching import evaluate_partial_match, normalize_sql


class HardnessPerformanceEvaluator:
    """Evaluates model performance grouped by SQL query hardness levels.
    
    The hardness classification is based ONLY on the gold (ground truth) SQL queries.
    The predicted queries are then evaluated within each hardness group.
    """
    
    def __init__(self, db_engine):
        self.classifier = SQLHardnessClassifier()
        self.db_engine = db_engine
        
    def group_by_hardness(self, gold_examples: List[Dict], pred_examples: List[Dict]) -> Dict[str, List[Tuple]]:
        """Group query pairs by hardness level of GOLD queries only.
        
        The hardness is determined solely by analyzing the gold SQL query.
        The predicted query is paired with it but does NOT affect the hardness classification.
        """
        hardness_groups = defaultdict(list)
        
        for gold, pred in zip(gold_examples, pred_examples):
            # Extract gold SQL - this is what determines hardness
            gold_sql = gold.get('query', gold.get('gold_sql', ''))
            
            # Classify hardness based ONLY on gold SQL
            hardness = self.classifier.classify_hardness(gold_sql)
            
            # Group the gold-pred pair by the gold query's hardness
            hardness_groups[hardness].append((gold, pred))
            
        return dict(hardness_groups)
    
    def show_hardness_distribution(self, gold_examples: List[Dict]) -> Dict[str, int]:
        """Show the distribution of hardness levels in gold queries.
        
        This method helps visualize how the gold queries are distributed
        across different hardness levels.
        """
        distribution = defaultdict(int)
        
        print("\n" + "="*60)
        print("GOLD QUERY HARDNESS CLASSIFICATION")
        print("="*60)
        
        for i, gold in enumerate(gold_examples):
            gold_sql = gold.get('query', gold.get('gold_sql', ''))
            hardness = self.classifier.classify_hardness(gold_sql)
            distribution[hardness] += 1
            # Show first few examples of each hardness level
            # if distribution[hardness] <= 2:  # Show first 2 examples of each level
            #     print(f"\nQuery {i+1} - {hardness.upper()}:")
            #     print(f"Question: {gold.get('question', 'N/A')}")
            #     print(f"SQL: {gold_sql[:100]}..." if len(gold_sql) > 100 else f"SQL: {gold_sql}")
        
        print("\n" + "-"*60)
        print("DISTRIBUTION SUMMARY:")
        total = len(gold_examples)
        for level in ['easy', 'medium', 'hard', 'extra']:
            count = distribution[level]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{level.upper():<10} {count:>3} queries ({percentage:>5.1f}%)")
        print("-"*60)
        
        return dict(distribution)
    
    def evaluate_execution_match(self, gold_sql: str, pred_sql: str) -> bool:
        """Check if two queries return the same results"""
        try:
            gold_df = pd.read_sql_query(gold_sql, self.db_engine)
            pred_df = pd.read_sql_query(pred_sql, self.db_engine)
            
            # Sort columns alphabetically
            gold_df = gold_df.reindex(sorted(gold_df.columns), axis=1)
            pred_df = pred_df.reindex(sorted(pred_df.columns), axis=1)
            
            # Sort rows
            gold_df = gold_df.sort_values(by=list(gold_df.columns)).reset_index(drop=True)
            pred_df = pred_df.sort_values(by=list(pred_df.columns)).reset_index(drop=True)
            
            # Check if values match (ignoring column names)
            if gold_df.shape == pred_df.shape:
                return (gold_df.values == pred_df.values).all()
            return False
            
        except Exception as e:
            return False
    
    def evaluate_by_hardness(self, gold_examples: List[Dict], pred_examples: List[Dict]) -> Dict[str, Dict]:
        """Evaluate performance metrics grouped by hardness level of GOLD queries.
        
        The evaluation process:
        1. Classify each gold query by hardness (easy/medium/hard/extra)
        2. Group gold-pred pairs by the gold query's hardness
        3. Evaluate predicted queries within each hardness group
        """
        # First, show the hardness distribution of gold queries
        self.show_hardness_distribution(gold_examples)
        
        # Group by hardness (based on gold queries only)
        hardness_groups = self.group_by_hardness(gold_examples, pred_examples)
        results = {}
        
        for hardness in ['easy', 'medium', 'hard', 'extra']:
            if hardness not in hardness_groups:
                results[hardness] = {
                    'count': 0,
                    'execution_accuracy': 0.0,
                    'partial_match': {},
                    'error_types': defaultdict(int)
                }
                continue
                
            group = hardness_groups[hardness]
            count = len(group)
            
            # Initialize metrics
            exec_matches = 0
            partial_scores = defaultdict(list)
            error_types = defaultdict(int)
            
            # Evaluate each query pair
            for gold, pred in group:
                gold_sql = gold.get('query', gold.get('gold_sql', ''))
                pred_sql = pred.get('query', pred.get('predicted_sql', ''))
                
                # Execution match
                exec_match = self.evaluate_execution_match(gold_sql, pred_sql)
                if exec_match:
                    exec_matches += 1
                else:
                    # Categorize error type
                    try:
                        pd.read_sql_query(pred_sql, self.db_engine)
                        error_types['wrong_result'] += 1
                    except Exception as e:
                        if 'syntax' in str(e).lower():
                            error_types['syntax_error'] += 1
                        elif 'column' in str(e).lower():
                            error_types['column_error'] += 1
                        elif 'table' in str(e).lower():
                            error_types['table_error'] += 1
                        else:
                            error_types['other_error'] += 1
                
                # Partial component matching
                partial_result = evaluate_partial_match(pred_sql, gold_sql)
                for component, scores in partial_result.items():
                    partial_scores[component].append(scores)
            
            # Calculate averages
            results[hardness] = {
                'count': count,
                'execution_accuracy': exec_matches / count if count > 0 else 0,
                'partial_match': {},
                'error_types': dict(error_types)
            }
            
            # Average partial match scores
            for component, score_list in partial_scores.items():
                results[hardness]['partial_match'][component] = {
                    'precision': mean([s['precision'] for s in score_list]),
                    'recall': mean([s['recall'] for s in score_list]),
                    'f1': mean([s['f1'] for s in score_list])
                }
                
        return results
    
    def print_performance_report(self, results: Dict[str, Dict], model_name: str = "Model"):
        """Print a formatted performance report by hardness level"""
        print(f"\n{'='*80}")
        print(f"PERFORMANCE REPORT BY HARDNESS LEVEL: {model_name}")
        print(f"{'='*80}\n")
        
        # Summary table
        print(f"{'Hardness':<10} {'Count':<8} {'Exec Acc':<12} {'Avg F1':<10}")
        print("-" * 50)
        
        total_count = 0
        total_exec_weighted = 0
        
        for hardness in ['easy', 'medium', 'hard', 'extra']:
            if hardness not in results or results[hardness]['count'] == 0:
                continue
                
            r = results[hardness]
            count = r['count']
            exec_acc = r['execution_accuracy']
            
            # Calculate average F1 across components
            avg_f1 = 0
            if r['partial_match']:
                f1_scores = [scores['f1'] for scores in r['partial_match'].values()]
                avg_f1 = mean(f1_scores) if f1_scores else 0
            
            print(f"{hardness:<10} {count:<8} {exec_acc:<12.2%} {avg_f1:<10.2%}")
            
            total_count += count
            total_exec_weighted += exec_acc * count
        
        # Overall weighted average
        if total_count > 0:
            print("-" * 50)
            overall_exec = total_exec_weighted / total_count
            print(f"{'OVERALL':<10} {total_count:<8} {overall_exec:<12.2%}")
        
        # Detailed breakdown for each hardness level
        print("\n" + "="*80)
        print("DETAILED BREAKDOWN BY HARDNESS LEVEL")
        print("="*80)
        
        for hardness in ['easy', 'medium', 'hard', 'extra']:
            if hardness not in results or results[hardness]['count'] == 0:
                continue
                
            r = results[hardness]
            print(f"\n{hardness.upper()} QUERIES (n={r['count']})")
            print("-" * 40)
            
            # Error analysis
            if r['error_types']:
                print("\nError Types:")
                for error_type, count in r['error_types'].items():
                    percentage = (count / r['count']) * 100
                    print(f"  {error_type:<20} {count:>3} ({percentage:>5.1f}%)")
            
            # Partial match scores
            if r['partial_match']:
                print("\nPartial Component Matching:")
                print(f"  {'Component':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
                print("  " + "-" * 45)
                for component, scores in r['partial_match'].items():
                    print(f"  {component:<15} {scores['precision']:<10.2%} {scores['recall']:<10.2%} {scores['f1']:<10.2%}")
    
    def compare_models(self, gold_file: str, model_files: Dict[str, str]):
        """Compare multiple models' performance by hardness level.
        
        Note: The hardness classification is based ONLY on the gold queries.
        All models are evaluated against the same hardness groupings.
        """
        with open(gold_file, 'r') as f:
            gold_examples = json.load(f)
        
        all_results = {}
        
        # Evaluate each model (without detailed output)
        for model_name, pred_file in model_files.items():
            try:
                with open(pred_file, 'r') as f:
                    pred_examples = json.load(f)
                
                # Group by hardness (based on gold queries only) - silent mode
                hardness_groups = self.group_by_hardness(gold_examples, pred_examples)
                results = {}
                
                for hardness in ['easy', 'medium', 'hard', 'extra']:
                    if hardness not in hardness_groups:
                        results[hardness] = {
                            'count': 0,
                            'execution_accuracy': 0.0,
                            'partial_match': {},
                            'error_types': defaultdict(int)
                        }
                        continue
                        
                    group = hardness_groups[hardness]
                    count = len(group)
                    
                    # Initialize metrics
                    exec_matches = 0
                    partial_scores = defaultdict(list)
                    error_types = defaultdict(int)
                    
                    # Evaluate each query pair
                    for gold, pred in group:
                        gold_sql = gold.get('query', gold.get('gold_sql', ''))
                        pred_sql = pred.get('query', pred.get('predicted_sql', ''))
                        
                        # Execution match
                        exec_match = self.evaluate_execution_match(gold_sql, pred_sql)
                        if exec_match:
                            exec_matches += 1
                        else:
                            # Categorize error type
                            try:
                                pd.read_sql_query(pred_sql, self.db_engine)
                                error_types['wrong_result'] += 1
                            except Exception as e:
                                if 'syntax' in str(e).lower():
                                    error_types['syntax_error'] += 1
                                elif 'column' in str(e).lower():
                                    error_types['column_error'] += 1
                                elif 'table' in str(e).lower():
                                    error_types['table_error'] += 1
                                else:
                                    error_types['other_error'] += 1
                        
                        # Partial component matching
                        partial_result = evaluate_partial_match(pred_sql, gold_sql)
                        for component, scores in partial_result.items():
                            partial_scores[component].append(scores)
                    
                    # Calculate averages
                    results[hardness] = {
                        'count': count,
                        'execution_accuracy': exec_matches / count if count > 0 else 0,
                        'partial_match': {},
                        'error_types': dict(error_types)
                    }
                    
                    # Average partial match scores
                    for component, score_list in partial_scores.items():
                        results[hardness]['partial_match'][component] = {
                            'precision': mean([s['precision'] for s in score_list]),
                            'recall': mean([s['recall'] for s in score_list]),
                            'f1': mean([s['f1'] for s in score_list])
                        }
                
                all_results[model_name] = results
                
            except FileNotFoundError:
                print(f"File {pred_file} not found. Skipping {model_name}...")
        
        # Print ONLY the comparison table
        self._print_comparison_report(all_results)
        
        return all_results
    
    def _print_comparison_report(self, all_results: Dict[str, Dict]):
        """Print a comparison report across multiple models"""
        print("\n" + "="*100)
        print("MODEL COMPARISON BY HARDNESS LEVEL")
        print("="*100)
        
        # Execution accuracy comparison ONLY
        print("\nEXECUTION ACCURACY:")
        print(f"{'Model':<30} {'Easy':<12} {'Medium':<12} {'Hard':<12} {'Extra':<12} {'Overall':<12}")
        print("-" * 90)
        
        for model_name, results in all_results.items():
            model_short = model_name[:28]
            row = f"{model_short:<30}"
            
            total_count = 0
            total_weighted = 0
            
            for hardness in ['easy', 'medium', 'hard', 'extra']:
                if hardness in results and results[hardness]['count'] > 0:
                    acc = results[hardness]['execution_accuracy']
                    count = results[hardness]['count']
                    row += f"{acc:<12.2%}"
                    total_count += count
                    total_weighted += acc * count
                else:
                    row += f"{'N/A':<12}"
            
            # Overall
            if total_count > 0:
                overall = total_weighted / total_count
                row += f"{overall:<12.2%}"
            else:
                row += f"{'N/A':<12}"
                
            print(row)


# Example usage
if __name__ == "__main__":
    # This would need to be run with your actual database connection
    # from sqlalchemy import create_engine
    # db_engine = create_engine("postgresql+psycopg2://user:pass@localhost:5432/db")
    
    # Example of how to use:
    """
    evaluator = HardnessPerformanceEvaluator(db_engine)
    
    # Evaluate a single model
    with open('gold_sql.json', 'r') as f:
        gold_examples = json.load(f)
    with open('generated_queries_XIYAN_SQL.json', 'r') as f:
        pred_examples = json.load(f)
    
    results = evaluator.evaluate_by_hardness(gold_examples, pred_examples)
    evaluator.print_performance_report(results, "XIYAN SQL")
    
    # Compare multiple models
    model_files = {
        'XIYAN_SQL': 'generated_queries_XIYAN_SQL.json',
        'SQL_Coder': 'generated_queries_SQL_Coder.json',
        'SQL_Coder_RAG': 'generated_queries_SQL_Coder_with_RAG.json'
    }
    
    all_results = evaluator.compare_models('gold_sql.json', model_files)
    """
    print("HardnessPerformanceEvaluator ready to use!") 