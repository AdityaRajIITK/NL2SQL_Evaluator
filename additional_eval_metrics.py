"""
Additional SQL Evaluation Metrics based on Spider evaluation.py

This module implements advanced evaluation metrics for SQL queries including:
1. Fine-grained component evaluation (with and without aggregations/operators)
2. AND/OR operator evaluation
3. HAVING clause evaluation
4. Set operations evaluation (INTERSECT/UNION/EXCEPT)
5. Nested query evaluation
6. SQL validity checking
7. Keyword presence evaluation

Each metric can be called individually or as part of a comprehensive evaluation.
"""

import re
import json
import pandas as pd
from typing import Dict, List, Tuple, Set, Union
from collections import defaultdict
from statistics import mean
import sqlparse


class AdvancedSQLEvaluator:
    """Advanced SQL evaluation metrics beyond basic execution match"""
    
    def __init__(self, db_engine=None):
        self.db_engine = db_engine
    
    # =============================================================================
    # VALIDITY METRICS
    # =============================================================================
    
    def check_sql_validity(self, sql: str) -> bool:
        """
        Check if SQL is syntactically valid using the database
        
        Args:
            sql: SQL query string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not self.db_engine:
            return True  # Can't check without database
            
        try:
            # Use EXPLAIN to check syntax without executing
            pd.read_sql_query(f"EXPLAIN {sql}", self.db_engine)
            return True
        except Exception:
            return False
    
    def evaluate_validity(self, pred_sql: str, gold_sql: str) -> Dict[str, bool]:
        """
        Evaluate validity of both predicted and gold SQL queries
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold/reference SQL query
            
        Returns:
            Dict with 'pred_valid' and 'gold_valid' boolean values
        """
        return {
            'pred_valid': self.check_sql_validity(pred_sql),
            'gold_valid': self.check_sql_validity(gold_sql)
        }
    
    # =============================================================================
    # SELECT CLAUSE EVALUATION
    # =============================================================================
        
    def evaluate_select_with_aggregations(self, pred_sql: str, gold_sql: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate SELECT clause with and without aggregations separately
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold/reference SQL query
            
        Returns:
            Dict with metrics for:
            - 'select_full': Full SELECT match (including aggregations)
            - 'select_no_agg': SELECT match without aggregations
            - 'aggregations_only': Aggregation functions only
        """
        # Extract SELECT columns
        pred_select = self._extract_select_items(pred_sql)
        gold_select = self._extract_select_items(gold_sql)
        
        # Separate aggregations from columns
        pred_aggs, pred_cols = self._separate_aggregations(pred_select)
        gold_aggs, gold_cols = self._separate_aggregations(gold_select)
        
        # Full match (with aggregations)
        full_match = self._calculate_match_metrics(pred_select, gold_select)
        
        # Column match (without aggregations)
        col_match = self._calculate_match_metrics(pred_cols, gold_cols)
        
        # Aggregation match
        agg_match = self._calculate_match_metrics(pred_aggs, gold_aggs)
        
        return {
            'select_full': full_match,
            'select_no_agg': col_match,
            'aggregations_only': agg_match
        }
    
    # =============================================================================
    # WHERE CLAUSE EVALUATION
    # =============================================================================
    
    def evaluate_where_with_operators(self, pred_sql: str, gold_sql: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate WHERE clause with and without operators
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold/reference SQL query
            
        Returns:
            Dict with metrics for:
            - 'where_full': Full WHERE conditions
            - 'where_no_op': WHERE conditions without operators
            - 'operators_only': Operator usage only
        """
        pred_where = self._extract_where_conditions(pred_sql)
        gold_where = self._extract_where_conditions(gold_sql)
        
        # Full conditions
        full_match = self._calculate_match_metrics(pred_where['full'], gold_where['full'])
        
        # Without operators
        no_op_match = self._calculate_match_metrics(pred_where['no_op'], gold_where['no_op'])
        
        # Operators only
        op_match = self._calculate_match_metrics(pred_where['operators'], gold_where['operators'])
        
        return {
            'where_full': full_match,
            'where_no_op': no_op_match,
            'operators_only': op_match
        }
    
    # =============================================================================
    # LOGICAL OPERATORS
    # =============================================================================
    
    def evaluate_and_or_usage(self, pred_sql: str, gold_sql: str) -> Dict[str, float]:
        """
        Evaluate AND/OR operator usage in WHERE clauses
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold/reference SQL query
            
        Returns:
            Dict with precision, recall, and F1 scores for logical operators
        """
        pred_ops = self._extract_logical_operators(pred_sql)
        gold_ops = self._extract_logical_operators(gold_sql)
        
        return self._calculate_match_metrics(pred_ops, gold_ops)
    
    # =============================================================================
    # HAVING CLAUSE EVALUATION
    # =============================================================================
    
    def evaluate_having_clause(self, pred_sql: str, gold_sql: str) -> Dict[str, float]:
        """
        Evaluate HAVING clause (requires GROUP BY to be meaningful)
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold/reference SQL query
            
        Returns:
            Dict with precision, recall, and F1 scores for HAVING clause
        """
        pred_has_having = 'having' in pred_sql.lower()
        gold_has_having = 'having' in gold_sql.lower()
        
        if not gold_has_having and not pred_has_having:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if gold_has_having != pred_has_having:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Extract GROUP BY columns
        pred_group = self._extract_group_by_columns(pred_sql)
        gold_group = self._extract_group_by_columns(gold_sql)
        
        # Extract HAVING conditions
        pred_having = self._extract_having_conditions(pred_sql)
        gold_having = self._extract_having_conditions(gold_sql)
        
        # Both must match
        group_match = set(pred_group) == set(gold_group)
        having_match = set(pred_having) == set(gold_having)
        
        if group_match and having_match:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        else:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # =============================================================================
    # SET OPERATIONS
    # =============================================================================
    
    def evaluate_set_operations(self, pred_sql: str, gold_sql: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate INTERSECT/UNION/EXCEPT operations
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold/reference SQL query
            
        Returns:
            Dict with metrics for each set operation (intersect, union, except)
        """
        set_ops = ['intersect', 'union', 'except']
        results = {}
        
        for op in set_ops:
            pred_has_op = op in pred_sql.lower()
            gold_has_op = op in gold_sql.lower()
            
            if pred_has_op == gold_has_op:
                if pred_has_op:
                    # Both have the operation - need to check the subqueries
                    results[op] = {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
                else:
                    # Neither has the operation
                    results[op] = {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
            else:
                results[op] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        return results
    
    # =============================================================================
    # NESTED QUERIES
    # =============================================================================
    
    def evaluate_nested_queries(self, pred_sql: str, gold_sql: str) -> Dict[str, float]:
        """
        Evaluate nested subqueries
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold/reference SQL query
            
        Returns:
            Dict with precision, recall, and F1 scores for nested queries
        """
        pred_nested_count = pred_sql.lower().count('select') - 1
        gold_nested_count = gold_sql.lower().count('select') - 1
        
        if pred_nested_count == gold_nested_count:
            if pred_nested_count == 0:
                return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
            else:
                # Has nested queries - simplified evaluation
                return {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
        else:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # =============================================================================
    # KEYWORD PRESENCE
    # =============================================================================
    
    def evaluate_keyword_presence(self, pred_sql: str, gold_sql: str) -> Dict[str, float]:
        """
        Evaluate presence of specific SQL keywords
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold/reference SQL query
            
        Returns:
            Dict with accuracy scores for each keyword (1.0 if both have/don't have, 0.0 otherwise)
        """
        keywords_to_check = {
            'where': r'\bwhere\b',
            'group': r'\bgroup\s+by\b',
            'having': r'\bhaving\b',
            'order': r'\border\s+by\b',
            'limit': r'\blimit\b',
            'distinct': r'\bdistinct\b',
            'join': r'\bjoin\b',
            'left_join': r'\bleft\s+join\b',
            'right_join': r'\bright\s+join\b',
            'inner_join': r'\binner\s+join\b',
            'not': r'\bnot\b',
            'in': r'\bin\b',
            'like': r'\blike\b',
            'between': r'\bbetween\b',
            'exists': r'\bexists\b'
        }
        
        results = {}
        for keyword, pattern in keywords_to_check.items():
            pred_has = bool(re.search(pattern, pred_sql, re.IGNORECASE))
            gold_has = bool(re.search(pattern, gold_sql, re.IGNORECASE))
            
            if pred_has == gold_has:
                results[keyword] = 1.0
            else:
                results[keyword] = 0.0
        
        return results
    
    # =============================================================================
    # ORDER BY AND LIMIT
    # =============================================================================
    
    def evaluate_order_with_limit(self, pred_sql: str, gold_sql: str) -> Dict[str, float]:
        """
        Evaluate ORDER BY considering LIMIT
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold/reference SQL query
            
        Returns:
            Dict with precision, recall, and F1 scores for ORDER BY/LIMIT
        """
        pred_has_order = 'order by' in pred_sql.lower()
        gold_has_order = 'order by' in gold_sql.lower()
        pred_has_limit = 'limit' in pred_sql.lower()
        gold_has_limit = 'limit' in gold_sql.lower()
        
        # If gold has ORDER BY, check full match
        if gold_has_order:
            if not pred_has_order:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            # Extract ORDER BY columns and directions
            pred_order = self._extract_order_by_details(pred_sql)
            gold_order = self._extract_order_by_details(gold_sql)
            
            # Check if LIMIT consistency
            if (gold_has_limit and not pred_has_limit) or (not gold_has_limit and pred_has_limit):
                return {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
            
            return self._calculate_match_metrics([pred_order], [gold_order])
        else:
            if pred_has_order:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            else:
                return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

    # =============================================================================
    # COMPREHENSIVE EVALUATION
    # =============================================================================
    
    def comprehensive_evaluation(self, pred_sql: str, gold_sql: str) -> Dict[str, any]:
        """
        Run all evaluation metrics
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold/reference SQL query
            
        Returns:
            Dict containing results from all evaluation metrics
        """
        results = {}
        
        # All individual evaluations
        results.update(self.evaluate_validity(pred_sql, gold_sql))
        results['select_evaluation'] = self.evaluate_select_with_aggregations(pred_sql, gold_sql)
        results['where_evaluation'] = self.evaluate_where_with_operators(pred_sql, gold_sql)
        results['and_or_evaluation'] = self.evaluate_and_or_usage(pred_sql, gold_sql)
        results['having_evaluation'] = self.evaluate_having_clause(pred_sql, gold_sql)
        results['order_limit_evaluation'] = self.evaluate_order_with_limit(pred_sql, gold_sql)
        results['set_operations'] = self.evaluate_set_operations(pred_sql, gold_sql)
        results['nested_queries'] = self.evaluate_nested_queries(pred_sql, gold_sql)
        results['keyword_presence'] = self.evaluate_keyword_presence(pred_sql, gold_sql)
        
        return results

    # =============================================================================
    # HELPER METHODS
    # =============================================================================

    def _extract_select_items(self, sql: str) -> List[str]:
        """Extract items from SELECT clause"""
        sql_lower = sql.lower()
        select_match = re.search(r'select\s+(.+?)\s+from', sql_lower, re.DOTALL)
        if not select_match:
            return []
        
        select_part = select_match.group(1)
        # Simple split by comma (doesn't handle nested functions perfectly)
        items = [item.strip() for item in select_part.split(',')]
        return items
    
    def _separate_aggregations(self, select_items: List[str]) -> Tuple[List[str], List[str]]:
        """Separate aggregation functions from regular columns"""
        agg_functions = ['max', 'min', 'count', 'sum', 'avg', 'group_concat']
        aggregations = []
        columns = []
        
        for item in select_items:
            has_agg = False
            for agg in agg_functions:
                if agg + '(' in item.lower():
                    aggregations.append(item)
                    has_agg = True
                    break
            if not has_agg:
                columns.append(item)
        
        return aggregations, columns
    
    def _extract_where_conditions(self, sql: str) -> Dict[str, List[str]]:
        """Extract WHERE conditions with and without operators"""
        sql_lower = sql.lower()
        where_match = re.search(r'where\s+(.+?)(?:group by|order by|having|limit|$)', 
                               sql_lower, re.DOTALL)
        if not where_match:
            return {'full': [], 'no_op': [], 'operators': []}
        
        where_clause = where_match.group(1).strip()
        
        # Extract conditions (simplified)
        conditions = re.split(r'\s+(?:and|or)\s+', where_clause)
        
        operators = ['=', '!=', '<>', '>', '<', '>=', '<=', 'like', 'in', 'between', 'is']
        
        full_conditions = conditions
        no_op_conditions = []
        used_operators = []
        
        for cond in conditions:
            # Extract operator
            for op in operators:
                if f' {op} ' in f' {cond} ':
                    used_operators.append(op)
                    # Remove operator for no_op version
                    parts = cond.split(op, 1)
                    if len(parts) > 0:
                        no_op_conditions.append(parts[0].strip())
                    break
        
        return {
            'full': full_conditions,
            'no_op': no_op_conditions,
            'operators': used_operators
        }
    
    def _extract_logical_operators(self, sql: str) -> List[str]:
        """Extract AND/OR operators from WHERE clause"""
        sql_lower = sql.lower()
        where_match = re.search(r'where\s+(.+?)(?:group by|order by|having|limit|$)', 
                               sql_lower, re.DOTALL)
        if not where_match:
            return []
        
        where_clause = where_match.group(1)
        operators = []
        
        # Find all AND/OR
        and_count = len(re.findall(r'\band\b', where_clause))
        or_count = len(re.findall(r'\bor\b', where_clause))
        
        operators.extend(['and'] * and_count)
        operators.extend(['or'] * or_count)
        
        return operators
    
    def _extract_group_by_columns(self, sql: str) -> List[str]:
        """Extract GROUP BY columns"""
        sql_lower = sql.lower()
        group_match = re.search(r'group\s+by\s+(.+?)(?:having|order by|limit|$)', 
                               sql_lower, re.DOTALL)
        if not group_match:
            return []
        
        group_part = group_match.group(1).strip()
        columns = [col.strip() for col in group_part.split(',')]
        return columns
    
    def _extract_having_conditions(self, sql: str) -> List[str]:
        """Extract HAVING conditions"""
        sql_lower = sql.lower()
        having_match = re.search(r'having\s+(.+?)(?:order by|limit|$)', 
                                sql_lower, re.DOTALL)
        if not having_match:
            return []
        
        having_part = having_match.group(1).strip()
        # Split by AND/OR
        conditions = re.split(r'\s+(?:and|or)\s+', having_part)
        return conditions
    
    def _extract_order_by_details(self, sql: str) -> str:
        """Extract ORDER BY columns and directions"""
        sql_lower = sql.lower()
        order_match = re.search(r'order\s+by\s+(.+?)(?:limit|$)', sql_lower, re.DOTALL)
        if not order_match:
            return ""
        
        return order_match.group(1).strip()
    
    def _calculate_match_metrics(self, pred_items: List[str], gold_items: List[str]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 for two lists"""
        if not pred_items and not gold_items:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if not pred_items or not gold_items:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        pred_set = set(item.lower().strip() for item in pred_items)
        gold_set = set(item.lower().strip() for item in gold_items)
        
        if not pred_set and not gold_set:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        intersection = pred_set & gold_set
        
        precision = len(intersection) / len(pred_set) if pred_set else 0
        recall = len(intersection) / len(gold_set) if gold_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}


# =============================================================================
# STANDALONE FUNCTIONS FOR INDIVIDUAL METRICS
# =============================================================================

def evaluate_sql_validity(pred_sql: str, gold_sql: str, db_engine=None) -> Dict[str, bool]:
    """
    Standalone function to evaluate SQL validity
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold/reference SQL query  
        db_engine: Database engine for validation (optional)
        
    Returns:
        Dict with validity status for both queries
    """
    evaluator = AdvancedSQLEvaluator(db_engine)
    return evaluator.evaluate_validity(pred_sql, gold_sql)


def print_validity_analysis(gold_file: str, pred_file: str, db_engine=None, show_sql: bool = False):
    """
    Standalone function to print complete validity analysis table
    
    Args:
        gold_file: Path to gold SQL file
        pred_file: Path to predicted SQL file
        db_engine: Database engine for validation (optional)
        show_sql: Whether to include SQL queries in the output (default: False)
    
    Usage:
        print_validity_analysis('gold_sql.json', 'generated_queries_XIYAN_SQL.json', db_engine)
    """
    print_validity_table(gold_file, pred_file, db_engine, show_sql)


def get_validity_dataframe(gold_file: str, pred_file: str, db_engine=None) -> pd.DataFrame:
    """
    Standalone function to get validity results as a DataFrame
    
    Args:
        gold_file: Path to gold SQL file
        pred_file: Path to predicted SQL file
        db_engine: Database engine for validation (optional)
        
    Returns:
        DataFrame with validity results for each query
    """
    return evaluate_validity_table(gold_file, pred_file, db_engine)


def evaluate_keyword_presence(pred_sql: str, gold_sql: str) -> Dict[str, float]:
    """
    Standalone function to evaluate keyword presence
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold/reference SQL query
        
    Returns:
        Dict with keyword presence scores
    """
    evaluator = AdvancedSQLEvaluator()
    return evaluator.evaluate_keyword_presence(pred_sql, gold_sql)


def evaluate_select_components(pred_sql: str, gold_sql: str) -> Dict[str, Dict[str, float]]:
    """
    Standalone function to evaluate SELECT clause components
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold/reference SQL query
        
    Returns:
        Dict with SELECT component evaluation results
    """
    evaluator = AdvancedSQLEvaluator()
    return evaluator.evaluate_select_with_aggregations(pred_sql, gold_sql)


def evaluate_where_components(pred_sql: str, gold_sql: str) -> Dict[str, Dict[str, float]]:
    """
    Standalone function to evaluate WHERE clause components
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold/reference SQL query
        
    Returns:
        Dict with WHERE component evaluation results
    """
    evaluator = AdvancedSQLEvaluator()
    return evaluator.evaluate_where_with_operators(pred_sql, gold_sql)


def evaluate_logical_operators(pred_sql: str, gold_sql: str) -> Dict[str, float]:
    """
    Standalone function to evaluate AND/OR operators
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold/reference SQL query
        
    Returns:
        Dict with logical operator evaluation results
    """
    evaluator = AdvancedSQLEvaluator()
    return evaluator.evaluate_and_or_usage(pred_sql, gold_sql)


def evaluate_having_clause(pred_sql: str, gold_sql: str) -> Dict[str, float]:
    """
    Standalone function to evaluate HAVING clause
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold/reference SQL query
        
    Returns:
        Dict with HAVING clause evaluation results
    """
    evaluator = AdvancedSQLEvaluator()
    return evaluator.evaluate_having_clause(pred_sql, gold_sql)


def evaluate_set_operations(pred_sql: str, gold_sql: str) -> Dict[str, Dict[str, float]]:
    """
    Standalone function to evaluate set operations (UNION, INTERSECT, EXCEPT)
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold/reference SQL query
        
    Returns:
        Dict with set operation evaluation results
    """
    evaluator = AdvancedSQLEvaluator()
    return evaluator.evaluate_set_operations(pred_sql, gold_sql)


def evaluate_nested_queries(pred_sql: str, gold_sql: str) -> Dict[str, float]:
    """
    Standalone function to evaluate nested queries
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold/reference SQL query
        
    Returns:
        Dict with nested query evaluation results
    """
    evaluator = AdvancedSQLEvaluator()
    return evaluator.evaluate_nested_queries(pred_sql, gold_sql)


def evaluate_order_limit(pred_sql: str, gold_sql: str) -> Dict[str, float]:
    """
    Standalone function to evaluate ORDER BY and LIMIT
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold/reference SQL query
        
    Returns:
        Dict with ORDER BY/LIMIT evaluation results
    """
    evaluator = AdvancedSQLEvaluator()
    return evaluator.evaluate_order_with_limit(pred_sql, gold_sql)


# =============================================================================
# BATCH EVALUATION FUNCTIONS
# =============================================================================

def evaluate_validity_table(gold_file: str, pred_file: str, db_engine=None) -> pd.DataFrame:
    """
    Evaluate SQL validity for all queries and return results in a table format
    
    Args:
        gold_file: Path to gold SQL file
        pred_file: Path to predicted SQL file
        db_engine: Database engine for validation (optional)
        
    Returns:
        DataFrame with validity results for each query
    """
    with open(gold_file, 'r') as f:
        gold_examples = json.load(f)
    with open(pred_file, 'r') as f:
        pred_examples = json.load(f)
    
    evaluator = AdvancedSQLEvaluator(db_engine)
    
    results = []
    for i, (gold, pred) in enumerate(zip(gold_examples, pred_examples)):
        gold_sql = gold.get('query', gold.get('gold_sql', ''))
        pred_sql = pred.get('query', pred.get('predicted_sql', ''))
        question = gold.get('question', f'Query {i+1}')
        
        # Check validity
        gold_valid = evaluator.check_sql_validity(gold_sql)
        pred_valid = evaluator.check_sql_validity(pred_sql)
        
        results.append({
            'Query_ID': i + 1,
            'Question': question[:50] + '...' if len(question) > 50 else question,
            'Gold_Valid': '✅' if gold_valid else '❌',
            'Pred_Valid': '✅' if pred_valid else '❌',
            'Both_Valid': '✅' if gold_valid and pred_valid else '❌',
            'Gold_SQL': gold_sql[:100] + '...' if len(gold_sql) > 100 else gold_sql,
            'Pred_SQL': pred_sql[:100] + '...' if len(pred_sql) > 100 else pred_sql
        })
    
    return pd.DataFrame(results)


def print_validity_table(gold_file: str, pred_file: str, db_engine=None, show_sql: bool = False):
    """
    Print a formatted validity table for all queries
    
    Args:
        gold_file: Path to gold SQL file
        pred_file: Path to predicted SQL file
        db_engine: Database engine for validation (optional)
        show_sql: Whether to include SQL queries in the output (default: False)
    """
    df = evaluate_validity_table(gold_file, pred_file, db_engine)
    
    print(f"\n{'='*80}")
    print("SQL QUERY VALIDITY ANALYSIS")
    print(f"{'='*80}\n")
    
    # Summary statistics
    total_queries = len(df)
    gold_valid_count = (df['Gold_Valid'] == '✅').sum()
    pred_valid_count = (df['Pred_Valid'] == '✅').sum()
    both_valid_count = (df['Both_Valid'] == '✅').sum()
    
    print("SUMMARY:")
    print(f"Total Queries: {total_queries}")
    print(f"Gold Valid: {gold_valid_count}/{total_queries} ({gold_valid_count/total_queries:.1%})")
    print(f"Predicted Valid: {pred_valid_count}/{total_queries} ({pred_valid_count/total_queries:.1%})")
    print(f"Both Valid: {both_valid_count}/{total_queries} ({both_valid_count/total_queries:.1%})")
    
    print(f"\n{'-'*80}")
    print("DETAILED RESULTS:")
    print(f"{'-'*80}")
    
    if show_sql:
        # Full table with SQL
        print(f"{'ID':<3} {'Question':<50} {'Gold':<6} {'Pred':<6} {'Both':<6}")
        print("-" * 80)
        for _, row in df.iterrows():
            print(f"{row['Query_ID']:<3} {row['Question']:<50} {row['Gold_Valid']:<6} {row['Pred_Valid']:<6} {row['Both_Valid']:<6}")
            print(f"    Gold SQL: {row['Gold_SQL']}")
            print(f"    Pred SQL: {row['Pred_SQL']}")
            print()
    else:
        # Compact table without SQL
        print(f"{'ID':<3} {'Question':<60} {'Gold':<6} {'Pred':<6} {'Both':<6}")
        print("-" * 80)
        for _, row in df.iterrows():
            print(f"{row['Query_ID']:<3} {row['Question']:<60} {row['Gold_Valid']:<6} {row['Pred_Valid']:<6} {row['Both_Valid']:<6}")
    
    # Show invalid queries if any
    invalid_gold = df[df['Gold_Valid'] == '❌']
    invalid_pred = df[df['Pred_Valid'] == '❌']
    
    if len(invalid_gold) > 0:
        print(f"\n{'-'*40}")
        print("INVALID GOLD QUERIES:")
        print(f"{'-'*40}")
        for _, row in invalid_gold.iterrows():
            print(f"Query {row['Query_ID']}: {row['Question']}")
            print(f"SQL: {row['Gold_SQL']}")
            print()
    
    if len(invalid_pred) > 0:
        print(f"\n{'-'*40}")
        print("INVALID PREDICTED QUERIES:")
        print(f"{'-'*40}")
        for _, row in invalid_pred.iterrows():
            print(f"Query {row['Query_ID']}: {row['Question']}")
            print(f"SQL: {row['Pred_SQL']}")
            print()


def evaluate_model_comprehensive(gold_file: str, pred_file: str, db_engine=None) -> Dict[str, any]:
    """Evaluate a model using all comprehensive metrics"""
    with open(gold_file, 'r') as f:
        gold_examples = json.load(f)
    with open(pred_file, 'r') as f:
        pred_examples = json.load(f)
    
    evaluator = AdvancedSQLEvaluator(db_engine)
    all_results = defaultdict(list)
    
    for gold, pred in zip(gold_examples, pred_examples):
        gold_sql = gold.get('query', gold.get('gold_sql', ''))
        pred_sql = pred.get('query', pred.get('predicted_sql', ''))
        
        result = evaluator.comprehensive_evaluation(pred_sql, gold_sql)
        
        # Aggregate results
        for key, value in result.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        all_results[f"{key}.{subkey}"].append(subvalue)
                    else:
                        all_results[f"{key}.{subkey}"].append(subvalue)
            else:
                all_results[key].append(value)
    
    # Calculate averages
    averaged_results = {}
    for key, values in all_results.items():
        if values and isinstance(values[0], dict):
            # Average dict values (precision, recall, f1)
            averaged_results[key] = {
                'precision': mean([v.get('precision', 0) for v in values]),
                'recall': mean([v.get('recall', 0) for v in values]),
                'f1': mean([v.get('f1', 0) for v in values])
            }
        else:
            # Average scalar values
            averaged_results[key] = mean([float(v) for v in values])
    
    return averaged_results


def evaluate_specific_metric(gold_file: str, pred_file: str, metric_name: str, db_engine=None) -> Union[Dict, float]:
    """
    Evaluate a specific metric across all examples in the files
    
    Args:
        gold_file: Path to gold SQL file
        pred_file: Path to predicted SQL file  
        metric_name: Name of metric to evaluate ('validity', 'keywords', 'select', 'where', etc.)
        db_engine: Database engine (required for validity check)
        
    Returns:
        Averaged results for the specified metric
    """
    with open(gold_file, 'r') as f:
        gold_examples = json.load(f)
    with open(pred_file, 'r') as f:
        pred_examples = json.load(f)
    
    evaluator = AdvancedSQLEvaluator(db_engine)
    
    # Map metric names to functions
    metric_functions = {
        'validity': evaluator.evaluate_validity,
        'keywords': evaluator.evaluate_keyword_presence,
        'select': evaluator.evaluate_select_with_aggregations,
        'where': evaluator.evaluate_where_with_operators,
        'logical_ops': evaluator.evaluate_and_or_usage,
        'having': evaluator.evaluate_having_clause,
        'set_ops': evaluator.evaluate_set_operations,
        'nested': evaluator.evaluate_nested_queries,
        'order_limit': evaluator.evaluate_order_with_limit
    }
    
    if metric_name not in metric_functions:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metric_functions.keys())}")
    
    metric_func = metric_functions[metric_name]
    all_results = []
    
    for gold, pred in zip(gold_examples, pred_examples):
        gold_sql = gold.get('query', gold.get('gold_sql', ''))
        pred_sql = pred.get('query', pred.get('predicted_sql', ''))
        
        result = metric_func(pred_sql, gold_sql)
        all_results.append(result)
    
    # Calculate averages based on result type
    if all_results and isinstance(all_results[0], dict):
        if all(isinstance(v, dict) for v in all_results[0].values()):
            # Nested dict (like select, where evaluations)
            averaged = {}
            for key in all_results[0].keys():
                averaged[key] = {
                    'precision': mean([r[key].get('precision', 0) for r in all_results]),
                    'recall': mean([r[key].get('recall', 0) for r in all_results]),
                    'f1': mean([r[key].get('f1', 0) for r in all_results])
                }
        else:
            # Single level dict (like keywords, validity)
            averaged = {}
            for key in all_results[0].keys():
                averaged[key] = mean([r[key] for r in all_results])
    else:
        # Single values
        averaged = mean(all_results)
    
    return averaged


def print_comprehensive_results(results: Dict[str, any], model_name: str = "Model"):
    """Print comprehensive evaluation results in a formatted way"""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION RESULTS: {model_name}")
    print(f"{'='*80}\n")
    
    # Group results by category
    categories = {
        'Validity': ['pred_valid', 'gold_valid'],
        'SELECT Evaluation': [k for k in results.keys() if k.startswith('select_evaluation')],
        'WHERE Evaluation': [k for k in results.keys() if k.startswith('where_evaluation')],
        'Logical Operators': [k for k in results.keys() if k.startswith('and_or_evaluation')],
        'HAVING Clause': [k for k in results.keys() if k.startswith('having_evaluation')],
        'ORDER BY/LIMIT': [k for k in results.keys() if k.startswith('order_limit_evaluation')],
        'Set Operations': [k for k in results.keys() if k.startswith('set_operations')],
        'Nested Queries': [k for k in results.keys() if k.startswith('nested_queries')],
        'Keyword Presence': [k for k in results.keys() if k.startswith('keyword_presence')]
    }
    
    for category, keys in categories.items():
        if not keys:
            continue
            
        print(f"\n{category}:")
        print("-" * 60)
        
        for key in keys:
            if key not in results:
                continue
                
            value = results[key]
            display_key = key.split('.')[-1] if '.' in key else key
            
            if isinstance(value, dict):
                print(f"  {display_key:<30} P: {value['precision']:.2%}  R: {value['recall']:.2%}  F1: {value['f1']:.2%}")
            else:
                print(f"  {display_key:<30} {value:.2%}")


if __name__ == "__main__":
    # Example usage for individual metrics
    print("Advanced SQL Evaluator - Individual Metrics Available!")
    print("\n" + "="*60)
    print("INDIVIDUAL METRIC FUNCTIONS:")
    print("="*60)
    
    examples = [
        "# Validity check (single queries)",
        "validity = evaluate_sql_validity(pred_sql, gold_sql, db_engine)",
        "",
        "# Validity analysis (complete table for all queries)",
        "print_validity_analysis('gold_sql.json', 'generated_queries_XIYAN_SQL.json', db_engine)",
        "",
        "# Get validity results as DataFrame",
        "df = get_validity_dataframe('gold_sql.json', 'generated_queries_XIYAN_SQL.json', db_engine)",
        "",
        "# Keyword presence",  
        "keywords = evaluate_keyword_presence(pred_sql, gold_sql)",
        "",
        "# SELECT components",
        "select_results = evaluate_select_components(pred_sql, gold_sql)",
        "",
        "# WHERE components", 
        "where_results = evaluate_where_components(pred_sql, gold_sql)",
        "",
        "# Logical operators",
        "logical_ops = evaluate_logical_operators(pred_sql, gold_sql)",
        "",
        "# HAVING clause",
        "having_results = evaluate_having_clause(pred_sql, gold_sql)",
        "",
        "# Set operations",
        "set_ops = evaluate_set_operations(pred_sql, gold_sql)",
        "",
        "# Nested queries", 
        "nested = evaluate_nested_queries(pred_sql, gold_sql)",
        "",
        "# ORDER BY/LIMIT",
        "order_limit = evaluate_order_limit(pred_sql, gold_sql)",
        "",
        "# Class-based approach",
        "evaluator = AdvancedSQLEvaluator(db_engine)",
        "results = evaluator.comprehensive_evaluation(pred_sql, gold_sql)",
        "",
        "# Specific metric for entire dataset",
        "keyword_results = evaluate_specific_metric('gold.json', 'pred.json', 'keywords')"
    ]
    
    for example in examples:
        print(example) 