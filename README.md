# NL2SQL Evaluator

A comprehensive evaluation framework for Natural Language to SQL (NL2SQL) models, providing advanced metrics and analysis tools for SQL query generation systems.

## 🚀 Features

### Core Evaluation Metrics
- **Execution Accuracy**: Tests if generated SQL produces correct results
- **Exact Match**: Checks for perfect SQL string matching
- **Component-level Evaluation**: Fine-grained analysis of SQL components
- **Partial Component Matching**: Sophisticated partial credit scoring

### Advanced Analysis
- **SQL Hardness Classification**: Categorizes queries by complexity (Easy/Medium/Hard/Extra Hard)
- **Performance by Difficulty**: Analyze model performance across different hardness levels
- **SQL Validity Checking**: Syntax validation using database engines
- **Keyword Presence Analysis**: Track usage of specific SQL keywords

### Supported SQL Features
- ✅ Basic SELECT queries
- ✅ JOINs and subqueries
- ✅ Aggregations (COUNT, SUM, AVG, etc.)
- ✅ GROUP BY and HAVING clauses
- ✅ ORDER BY and LIMIT
- ✅ Set operations (UNION, INTERSECT, EXCEPT)
- ✅ Complex WHERE conditions with AND/OR
- ✅ Nested queries

## 📁 Project Structure

```
├── evaluation.py                      # Core Spider-based evaluation metrics
├── additional_eval_metrics.py         # Advanced evaluation metrics
├── hardness_performance_evaluator.py  # SQL difficulty analysis
├── sql_hardness_classifier.py         # SQL complexity classification
├── partial_component_matching.py      # Partial credit evaluation
├── process_sql.py                     # SQL parsing utilities
├── evaluator.ipynb                    # Jupyter notebook for analysis
├── reqirements.txt                    # Python dependencies
└── *.json                            # Sample query datasets
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/AdityaRajIITK/NL2SQL_Evaluator.git
cd NL2SQL_Evaluator
```

2. **Install dependencies**:
```bash
pip install -r reqirements.txt
```

3. **Set up database** (if using execution accuracy):
   - Ensure you have access to the target database
   - Configure database connection parameters

## 📊 Usage

### Basic Evaluation

```python
from evaluation import evaluate
from additional_eval_metrics import AdvancedSQLEvaluator

# Load your data
gold_queries = []  # List of ground truth SQL queries
pred_queries = []  # List of generated SQL queries

# Basic Spider evaluation
results = evaluate(gold=gold_queries, pred=pred_queries, db_dir="path/to/db")

# Advanced metrics
evaluator = AdvancedSQLEvaluator(db_engine=your_db_engine)
advanced_results = evaluator.comprehensive_evaluation(pred_queries, gold_queries)
```

### Hardness-based Analysis

```python
from hardness_performance_evaluator import HardnessPerformanceEvaluator

evaluator = HardnessPerformanceEvaluator(db_engine=your_db_engine)
hardness_results = evaluator.evaluate_by_hardness(gold_examples, pred_examples)

# Get performance breakdown by difficulty
performance_summary = evaluator.get_performance_summary(hardness_results)
```

### Component-level Evaluation

```python
from partial_component_matching import evaluate_partial_match

# Detailed component analysis
component_results = evaluate_partial_match(
    predicted_sql=pred_query,
    gold_sql=gold_query,
    db_engine=your_db_engine
)
```

## 📈 Evaluation Metrics

### 1. **Execution Accuracy**
- Tests whether generated SQL produces the same results as gold SQL
- Most reliable metric for practical applications

### 2. **Exact Match**
- Perfect string matching after normalization
- Strict but doesn't account for equivalent SQL variations

### 3. **Component Scores**
- **SELECT**: Column selection accuracy
- **FROM**: Table and join accuracy  
- **WHERE**: Condition accuracy
- **GROUP BY**: Grouping column accuracy
- **ORDER BY**: Sorting specification accuracy
- **Keywords**: Presence of required SQL keywords

### 4. **Hardness Classification**
- **Easy**: Simple SELECT with basic WHERE
- **Medium**: JOINs, GROUP BY, basic aggregations
- **Hard**: Nested queries, complex conditions
- **Extra Hard**: Multiple subqueries, advanced set operations

## 📝 Data Format

### Input Format
Your JSON files should contain:
```json
{
  "query": "SELECT name FROM students WHERE age > 18",
  "gold_sql": "SELECT name FROM students WHERE age > 18",
  "db_id": "school",
  "question": "What are the names of students older than 18?"
}
```

### Sample Files Included
- `generated_queries_SQL_Coder.json`
- `generated_queries_SQL_Coder_with_RAG.json`
- `generated_queries_XIYAN_SQL.json`
- `gold_sql.json`

## 🔧 Configuration

### Database Setup
```python
import sqlalchemy

# Example for SQLite
engine = sqlalchemy.create_engine('sqlite:///path/to/database.db')

# Example for PostgreSQL
engine = sqlalchemy.create_engine('postgresql://user:pass@localhost/dbname')
```

### Evaluation Flags
```python
# Disable value evaluation for faster processing
DISABLE_VALUE = True

# Disable distinct checking
DISABLE_DISTINCT = True
```

## 📊 Example Output

```
Overall Results:
- Execution Accuracy: 85.2%
- Exact Match: 72.1%
- Component Match: 78.9%

By Hardness Level:
- Easy: 92.3% execution accuracy
- Medium: 81.7% execution accuracy  
- Hard: 68.4% execution accuracy
- Extra Hard: 45.2% execution accuracy

Component Breakdown:
- SELECT: 89.1%
- FROM: 85.6%
- WHERE: 73.2%
- GROUP BY: 67.8%
- ORDER BY: 71.4%
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Built upon the [Spider](https://yale-lily.github.io/spider) evaluation framework
- Extends Spider metrics as it can handle more complex sql queries which have alias and custom column name in output

**Happy Evaluating! 🎯** 
