import re
import json
import sqlparse
from typing import List, Dict, Tuple, Set
from statistics import mean

# SQL component keywords
CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group by', 'having', 'order by', 'limit')
AGG_OPS = ('max', 'min', 'count', 'sum', 'avg', 'lower', 'upper')
WHERE_OPS = ('=', '>', '<', '>=', '<=', '!=', '<>', 'in', 'like', 'between', 'is', 'not')

def normalize_sql(sql: str) -> str:
    """Normalize SQL: lowercase, remove extra spaces, strip semicolon"""
    sql = sql.strip().rstrip(';')
    sql = re.sub(r'\s+', ' ', sql)
    return sql.lower()

def extract_tables(sql: str) -> Set[str]:
    """Extract table names from SQL"""
    parsed = sqlparse.parse(sql)[0]
    tables = set()
    from_seen = False
    for token in parsed.tokens:
        if from_seen:
            if isinstance(token, sqlparse.sql.IdentifierList):
                for identifier in token.get_identifiers():
                    tables.add(str(identifier).strip().lower())
            elif isinstance(token, sqlparse.sql.Identifier):
                tables.add(str(token).strip().lower())
            elif token.ttype is None:
                tables.add(str(token).strip().lower())
        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
            from_seen = True
        elif token.ttype is sqlparse.tokens.Keyword and token.value.upper() in ('WHERE', 'GROUP BY', 'ORDER BY', 'HAVING'):
            from_seen = False
    return tables

def extract_select_columns(sql: str) -> List[str]:
    """Extract columns from SELECT clause"""
    sql_norm = normalize_sql(sql)
    select_match = re.search(r'select\s+(.+?)\s+from', sql_norm, re.IGNORECASE)
    if not select_match:
        return []
    
    select_part = select_match.group(1)
    columns = [col.strip() for col in select_part.split(',')]
    return columns

def extract_where_conditions(sql: str) -> List[str]:
    """Extract WHERE conditions"""
    sql_norm = normalize_sql(sql)
    where_match = re.search(r'where\s+(.+?)(?:group by|order by|having|limit|$)', sql_norm, re.IGNORECASE)
    if not where_match:
        return []
    
    where_part = where_match.group(1)
    # Split by AND/OR while preserving them
    conditions = re.split(r'\s+(and|or)\s+', where_part, flags=re.IGNORECASE)
    return [cond.strip() for cond in conditions]

def extract_group_by(sql: str) -> List[str]:
    """Extract GROUP BY columns"""
    sql_norm = normalize_sql(sql)
    group_match = re.search(r'group by\s+(.+?)(?:having|order by|limit|$)', sql_norm, re.IGNORECASE)
    if not group_match:
        return []
    
    group_part = group_match.group(1)
    columns = [col.strip() for col in group_part.split(',')]
    return columns

def extract_order_by(sql: str) -> Tuple[List[str], List[str]]:
    """Extract ORDER BY columns and directions"""
    sql_norm = normalize_sql(sql)
    order_match = re.search(r'order by\s+(.+?)(?:limit|$)', sql_norm, re.IGNORECASE)
    if not order_match:
        return [], []
    
    order_part = order_match.group(1)
    columns = []
    directions = []
    
    for item in order_part.split(','):
        parts = item.strip().split()
        columns.append(parts[0])
        if len(parts) > 1 and parts[1].upper() in ('ASC', 'DESC'):
            directions.append(parts[1].upper())
        else:
            directions.append('ASC')  # Default
    
    return columns, directions

def extract_aggregations(sql: str) -> List[str]:
    """Extract aggregation functions"""
    sql_norm = normalize_sql(sql)
    aggs = []
    for agg in AGG_OPS:
        pattern = rf'{agg}\s*\('
        matches = re.findall(pattern, sql_norm, re.IGNORECASE)
        aggs.extend([agg] * len(matches))
    return aggs

def has_limit(sql: str) -> bool:
    """Check if SQL has LIMIT clause"""
    return bool(re.search(r'\blimit\b', normalize_sql(sql), re.IGNORECASE))

def has_distinct(sql: str) -> bool:
    """Check if SQL has DISTINCT"""
    return bool(re.search(r'\bdistinct\b', normalize_sql(sql), re.IGNORECASE))

def eval_component_match(pred_items: List[str], gold_items: List[str]) -> Tuple[float, float, float]:
    """Evaluate component matching between predicted and gold items"""
    if not gold_items and not pred_items:
        return 1.0, 1.0, 1.0
    
    if not gold_items or not pred_items:
        return 0.0, 0.0, 0.0
    
    pred_set = set(item.lower() for item in pred_items)
    gold_set = set(item.lower() for item in gold_items)
    
    matches = pred_set & gold_set
    
    precision = len(matches) / len(pred_set) if pred_set else 0
    recall = len(matches) / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate_partial_match(pred_sql: str, gold_sql: str) -> Dict[str, Dict[str, float]]:
    """Evaluate partial matching for all SQL components"""
    results = {}
    
    # SELECT columns
    pred_select = extract_select_columns(pred_sql)
    gold_select = extract_select_columns(gold_sql)
    precision, recall, f1 = eval_component_match(pred_select, gold_select)
    results['select'] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    # Tables
    pred_tables = list(extract_tables(pred_sql))
    gold_tables = list(extract_tables(gold_sql))
    precision, recall, f1 = eval_component_match(pred_tables, gold_tables)
    results['tables'] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    # WHERE conditions
    pred_where = extract_where_conditions(pred_sql)
    gold_where = extract_where_conditions(gold_sql)
    # Filter out AND/OR operators for condition matching
    pred_where_filtered = [w for w in pred_where if w.lower() not in ('and', 'or')]
    gold_where_filtered = [w for w in gold_where if w.lower() not in ('and', 'or')]
    precision, recall, f1 = eval_component_match(pred_where_filtered, gold_where_filtered)
    results['where'] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    # GROUP BY
    pred_group = extract_group_by(pred_sql)
    gold_group = extract_group_by(gold_sql)
    precision, recall, f1 = eval_component_match(pred_group, gold_group)
    results['group_by'] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    # ORDER BY
    pred_order, pred_dirs = extract_order_by(pred_sql)
    gold_order, gold_dirs = extract_order_by(gold_sql)
    precision, recall, f1 = eval_component_match(pred_order, gold_order)
    results['order_by'] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    # Aggregations
    pred_aggs = extract_aggregations(pred_sql)
    gold_aggs = extract_aggregations(gold_sql)
    precision, recall, f1 = eval_component_match(pred_aggs, gold_aggs)
    results['aggregations'] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    # Keywords (LIMIT, DISTINCT)
    pred_keywords = []
    gold_keywords = []
    if has_limit(pred_sql): pred_keywords.append('limit')
    if has_distinct(pred_sql): pred_keywords.append('distinct')
    if has_limit(gold_sql): gold_keywords.append('limit')
    if has_distinct(gold_sql): gold_keywords.append('distinct')
    precision, recall, f1 = eval_component_match(pred_keywords, gold_keywords)
    results['keywords'] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    return results

def evaluate_all_partial_matches(gold_examples: List[Dict], pred_examples: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Evaluate partial matches for all examples"""
    all_results = {}
    components = ['select', 'tables', 'where', 'group_by', 'order_by', 'aggregations', 'keywords']
    
    for component in components:
        all_results[component] = {'precision': [], 'recall': [], 'f1': []}
    
    for gold, pred in zip(gold_examples, pred_examples):
        gold_sql = gold.get('query', gold.get('gold_sql', ''))
        pred_sql = pred.get('query', pred.get('predicted_sql', ''))
        
        partial_results = evaluate_partial_match(pred_sql, gold_sql)
        
        for component in components:
            if component in partial_results:
                all_results[component]['precision'].append(partial_results[component]['precision'])
                all_results[component]['recall'].append(partial_results[component]['recall'])
                all_results[component]['f1'].append(partial_results[component]['f1'])
    
    # Calculate averages
    avg_results = {}
    for component in components:
        avg_results[component] = {
            'precision': mean(all_results[component]['precision']) if all_results[component]['precision'] else 0,
            'recall': mean(all_results[component]['recall']) if all_results[component]['recall'] else 0,
            'f1': mean(all_results[component]['f1']) if all_results[component]['f1'] else 0
        }
    
    return avg_results

if __name__ == "__main__":
    # Load your data
    with open('gold_sql.json', 'r') as f:
        gold_examples = json.load(f)
    
    # You can test with different generated query files
    generated_files = [
        'generated_queries_XIYAN_SQL.json',
        'generated_queries_SQL_Coder.json',
        'generated_queries_SQL_Coder_with_RAG.json'
    ]
    
    for gen_file in generated_files:
        print(f"\n{'='*70}")
        print(f"Evaluating: {gen_file}")
        print('='*70)
        
        with open(gen_file, 'r') as f:
            pred_examples = json.load(f)
        
        # Evaluate partial matches
        partial_results = evaluate_all_partial_matches(gold_examples, pred_examples)
        
        # Print results
        print("\n==================== PARTIAL COMPONENT MATCHING ====================\n")
        print(f"{'Component':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 55)
        for component, scores in partial_results.items():
            print(f"{component:<15} {scores['precision']:<12.3f} {scores['recall']:<12.3f} {scores['f1']:<12.3f}")
        
        # Overall average
        overall_precision = mean([scores['precision'] for scores in partial_results.values()])
        overall_recall = mean([scores['recall'] for scores in partial_results.values()])
        overall_f1 = mean([scores['f1'] for scores in partial_results.values()])
        print("-" * 55)
        print(f"{'OVERALL':<15} {overall_precision:<12.3f} {overall_recall:<12.3f} {overall_f1:<12.3f}") 