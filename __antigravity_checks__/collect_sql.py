import os
import re

def extract_sql_queries(start_dir):
    sql_pattern = re.compile(r'cur\.execute\(\s*"""(.*?)"""|cur\.execute\(\s*\'\'\'(.*?)\'\'\'|cur\.execute\(\s*"(.*?)"|cur\.execute\(\s*\'(.*?)\'', re.DOTALL)
    results = {}
    
    for root, dirs, files in os.walk(start_dir):
        if '__anygravity_checks__' in root or '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    matches = sql_pattern.findall(content)
                    if matches:
                        unique_queries = []
                        for m in matches:
                            query = next(filter(None, m)).strip()
                            if query not in unique_queries:
                                unique_queries.append(query)
                        results[path] = unique_queries
    return results

if __name__ == "__main__":
    queries = extract_sql_queries('.')
    with open('__antigravity_checks__/sql_baseline.txt', 'w', encoding='utf-8') as f:
        for path, qs in queries.items():
            f.write(f"FILE: {path}\n")
            for q in qs:
                f.write(f"  QUERY: {q}\n")
            f.write("-" * 40 + "\n")
