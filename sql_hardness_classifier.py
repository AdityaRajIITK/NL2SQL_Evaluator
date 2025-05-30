import re
import json
from typing import Dict, List, Tuple
from collections import Counter

class SQLHardnessClassifier:
    """
    Classifies SQL queries into difficulty levels: easy, medium, hard, extra
    Based on the complexity of SQL components and features used.
    """
    
    def __init__(self):
        # Component categories for hardness evaluation
        self.COMPONENT1_FEATURES = {
            'where': self._has_where,
            'group': self._has_group_by,
            'order': self._has_order_by,
            'limit': self._has_limit,
            'join': self._has_join,
            'or': self._has_or_operator,
            'like': self._has_like_operator
        }
        
        self.COMPONENT2_FEATURES = {
            'except': self._has_except,
            'union': self._has_union,
            'intersect': self._has_intersect,
            'nested': self._has_nested_query
        }
        
        self.OTHER_FEATURES = {
            'multiple_aggregations': self._has_multiple_aggregations,
            'multiple_select_columns': self._has_multiple_select_columns,
            'multiple_where_conditions': self._has_multiple_where_conditions,
            'multiple_group_by': self._has_multiple_group_by
        }
    
    def normalize_sql(self, sql: str) -> str:
        """Normalize SQL for analysis"""
        sql = sql.strip().rstrip(';')
        sql = re.sub(r'\s+', ' ', sql)
        return sql
    
    def _has_where(self, sql: str) -> bool:
        """Check if query has WHERE clause"""
        return bool(re.search(r'\bwhere\b', sql, re.IGNORECASE))
    
    def _has_group_by(self, sql: str) -> bool:
        """Check if query has GROUP BY clause"""
        return bool(re.search(r'\bgroup\s+by\b', sql, re.IGNORECASE))
    
    def _has_order_by(self, sql: str) -> bool:
        """Check if query has ORDER BY clause"""
        return bool(re.search(r'\border\s+by\b', sql, re.IGNORECASE))
    
    def _has_limit(self, sql: str) -> bool:
        """Check if query has LIMIT clause"""
        return bool(re.search(r'\blimit\b', sql, re.IGNORECASE))
    
    def _has_join(self, sql: str) -> bool:
        """Check if query has JOIN operations"""
        join_patterns = [
            r'\bjoin\b',
            r'\binner\s+join\b',
            r'\bleft\s+join\b',
            r'\bright\s+join\b',
            r'\bfull\s+join\b',
            r'\bcross\s+join\b'
        ]
        return any(re.search(pattern, sql, re.IGNORECASE) for pattern in join_patterns)
    
    def _has_or_operator(self, sql: str) -> bool:
        """Check if query has OR operators in conditions"""
        # Look for OR in WHERE clause
        where_match = re.search(r'where\s+(.+?)(?:group\s+by|order\s+by|having|limit|$)', sql, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)
            return bool(re.search(r'\bor\b', where_clause, re.IGNORECASE))
        return False
    
    def _has_like_operator(self, sql: str) -> bool:
        """Check if query has LIKE operator"""
        return bool(re.search(r'\blike\b', sql, re.IGNORECASE))
    
    def _has_except(self, sql: str) -> bool:
        """Check if query has EXCEPT operator"""
        return bool(re.search(r'\bexcept\b', sql, re.IGNORECASE))
    
    def _has_union(self, sql: str) -> bool:
        """Check if query has UNION operator"""
        return bool(re.search(r'\bunion\b', sql, re.IGNORECASE))
    
    def _has_intersect(self, sql: str) -> bool:
        """Check if query has INTERSECT operator"""
        return bool(re.search(r'\bintersect\b', sql, re.IGNORECASE))
    
    def _has_nested_query(self, sql: str) -> bool:
        """Check if query has nested subqueries"""
        # Count opening and closing parentheses after SELECT
        select_pos = sql.lower().find('select')
        if select_pos >= 0:
            after_select = sql[select_pos:]
            # Look for SELECT within parentheses
            return bool(re.search(r'\(\s*select\b', after_select, re.IGNORECASE))
        return False
    
    def _count_aggregations(self, sql: str) -> int:
        """Count number of aggregation functions"""
        agg_functions = ['max', 'min', 'count', 'sum', 'avg']
        count = 0
        for agg in agg_functions:
            pattern = rf'\b{agg}\s*\('
            count += len(re.findall(pattern, sql, re.IGNORECASE))
        return count
    
    def _has_multiple_aggregations(self, sql: str) -> bool:
        """Check if query has more than one aggregation"""
        return self._count_aggregations(sql) > 1
    
    def _count_select_columns(self, sql: str) -> int:
        """Count number of columns in SELECT clause"""
        select_match = re.search(r'select\s+(.+?)\s+from', sql, re.IGNORECASE)
        if select_match:
            select_part = select_match.group(1)
            # Remove aggregations to avoid double counting
            select_part = re.sub(r'\b(?:max|min|count|sum|avg)\s*\([^)]+\)', 'AGG', select_part, flags=re.IGNORECASE)
            # Count commas + 1
            return select_part.count(',') + 1
        return 0
    
    def _has_multiple_select_columns(self, sql: str) -> bool:
        """Check if query selects more than one column"""
        return self._count_select_columns(sql) > 1
    
    def _count_where_conditions(self, sql: str) -> int:
        """Count number of conditions in WHERE clause"""
        where_match = re.search(r'where\s+(.+?)(?:group\s+by|order\s+by|having|limit|$)', sql, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)
            # Count AND/OR operators + 1
            and_count = len(re.findall(r'\band\b', where_clause, re.IGNORECASE))
            or_count = len(re.findall(r'\bor\b', where_clause, re.IGNORECASE))
            return and_count + or_count + 1
        return 0
    
    def _has_multiple_where_conditions(self, sql: str) -> bool:
        """Check if query has more than one WHERE condition"""
        return self._count_where_conditions(sql) > 1
    
    def _count_group_by_columns(self, sql: str) -> int:
        """Count number of columns in GROUP BY clause"""
        group_match = re.search(r'group\s+by\s+(.+?)(?:having|order\s+by|limit|$)', sql, re.IGNORECASE)
        if group_match:
            group_part = group_match.group(1)
            # Count commas + 1
            return group_part.count(',') + 1
        return 0
    
    def _has_multiple_group_by(self, sql: str) -> bool:
        """Check if query groups by more than one column"""
        return self._count_group_by_columns(sql) > 1
    
    def count_component1(self, sql: str) -> int:
        """Count Component 1 features (basic SQL features)"""
        sql = self.normalize_sql(sql)
        count = 0
        
        for feature_name, check_func in self.COMPONENT1_FEATURES.items():
            if check_func(sql):
                count += 1
                
        # Special case: count multiple JOINs
        join_count = len(re.findall(r'\bjoin\b', sql, re.IGNORECASE))
        if join_count > 1:
            count += join_count - 1
            
        return count
    
    def count_component2(self, sql: str) -> int:
        """Count Component 2 features (advanced SQL features)"""
        sql = self.normalize_sql(sql)
        count = 0
        
        for feature_name, check_func in self.COMPONENT2_FEATURES.items():
            if check_func(sql):
                count += 1
                
        return count
    
    def count_others(self, sql: str) -> int:
        """Count other complexity features"""
        sql = self.normalize_sql(sql)
        count = 0
        
        for feature_name, check_func in self.OTHER_FEATURES.items():
            if check_func(sql):
                count += 1
                
        return count
    
    def classify_hardness(self, sql: str) -> str:
        """
        Classify SQL query hardness based on component counts.
        Returns: 'easy', 'medium', 'hard', or 'extra'
        """
        comp1 = self.count_component1(sql)
        comp2 = self.count_component2(sql)
        others = self.count_others(sql)
        
        # Classification rules based on evaluation.py
        if comp1 <= 1 and others == 0 and comp2 == 0:
            return "easy"
        elif (others <= 2 and comp1 <= 1 and comp2 == 0) or \
             (comp1 <= 2 and others < 2 and comp2 == 0):
            return "medium"
        elif (others > 2 and comp1 <= 2 and comp2 == 0) or \
             (2 < comp1 <= 3 and others <= 2 and comp2 == 0) or \
             (comp1 <= 1 and others == 0 and comp2 <= 1):
            return "hard"
        else:
            return "extra"
    
    def get_detailed_analysis(self, sql: str) -> Dict:
        """Get detailed analysis of SQL complexity"""
        sql = self.normalize_sql(sql)
        
        analysis = {
            'hardness': self.classify_hardness(sql),
            'component1_count': self.count_component1(sql),
            'component2_count': self.count_component2(sql),
            'others_count': self.count_others(sql),
            'features': {
                'component1': {},
                'component2': {},
                'others': {}
            }
        }
        
        # Check each feature
        for feature_name, check_func in self.COMPONENT1_FEATURES.items():
            analysis['features']['component1'][feature_name] = check_func(sql)
            
        for feature_name, check_func in self.COMPONENT2_FEATURES.items():
            analysis['features']['component2'][feature_name] = check_func(sql)
            
        for feature_name, check_func in self.OTHER_FEATURES.items():
            analysis['features']['others'][feature_name] = check_func(sql)
            
        return analysis


def classify_queries_from_json(json_file: str) -> List[Dict]:
    """Classify all queries in a JSON file"""
    classifier = SQLHardnessClassifier()
    
    with open(json_file, 'r') as f:
        queries = json.load(f)
    
    results = []
    for item in queries:
        sql = item.get('query', item.get('gold_sql', ''))
        analysis = classifier.get_detailed_analysis(sql)
        
        result = {
            'question': item.get('question', ''),
            'sql': sql,
            'hardness': analysis['hardness'],
            'component_counts': {
                'component1': analysis['component1_count'],
                'component2': analysis['component2_count'],
                'others': analysis['others_count']
            }
        }
        results.append(result)
    
    return results


def print_hardness_distribution(results: List[Dict]):
    """Print distribution of query hardness levels"""
    hardness_counts = Counter(r['hardness'] for r in results)
    total = len(results)
    
    print("\n==================== HARDNESS DISTRIBUTION ====================")
    print(f"Total queries: {total}\n")
    
    for level in ['easy', 'medium', 'hard', 'extra']:
        count = hardness_counts.get(level, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{level.upper():<10} {count:>3} queries ({percentage:>5.1f}%)")
    
    print("\n===============================================================")


if __name__ == "__main__":
    # Example usage
    classifier = SQLHardnessClassifier()
    
    # Test with your JSON files
    json_files = [
        'gold_sql.json',
        'generated_queries_XIYAN_SQL.json',
        'generated_queries_SQL_Coder.json',
        'generated_queries_SQL_Coder_with_RAG.json'
    ]
    
    for json_file in json_files:
        try:
            print(f"\n\nAnalyzing: {json_file}")
            print("="*70)
            
            results = classify_queries_from_json(json_file)
            print_hardness_distribution(results)
            
            # Show a few examples
            print("\nExample classifications:")
            for i, result in enumerate(results[:3]):
                print(f"\nQuery {i+1}:")
                print(f"Question: {result['question']}")
                print(f"SQL: {result['sql']}")
                print(f"Hardness: {result['hardness']}")
                print(f"Components: C1={result['component_counts']['component1']}, "
                      f"C2={result['component_counts']['component2']}, "
                      f"Others={result['component_counts']['others']}")
                
        except FileNotFoundError:
            print(f"File {json_file} not found. Skipping...") 