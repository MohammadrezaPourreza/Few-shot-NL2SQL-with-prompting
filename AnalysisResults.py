import subprocess
import os
import pandas as pd
from tqdm import tqdm

def load_results(directory):
    with open(directory+"/predicted_SQLS.txt", 'r') as file:
        predicted_sqls = file.readlines()
    with open(directory+"/Gold_SQLS.txt", 'r') as file:
        gold_sqls = file.readlines()
    sqls = []
    for gold_sql,predicted_sql in zip(gold_sqls,predicted_sqls):
        sqls.append([gold_sql.split("\t")[1].strip(),gold_sql.split("\t")[0].strip(),predicted_sql.strip()])
    return sqls

def get_accuracy(db_id,gold_sql,predicted_sql):
    with open('test-suite-sql-eval-master/Gold_test.txt', 'w') as f:
        f.write(gold_sql + "\t" + db_id)
    with open('test-suite-sql-eval-master/Predicted_test.txt', 'w') as f:
        f.write(predicted_sql)
    cmd_str = "python3 test-suite-sql-eval-master/evaluation.py --gold test-suite-sql-eval-master/Gold_test.txt --pred test-suite-sql-eval-master/Predicted_test.txt --db test-suite-sql-eval-master/database/ --etype exec "
    result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    os.remove("test-suite-sql-eval-master/Gold_test.txt")
    os.remove("test-suite-sql-eval-master/Predicted_test.txt")
    acc = float(result.stdout[-21:-16])
    return acc

if __name__ == '__main__':
    first_directory = "Results/SQLChainWithSchema_linking"
    second_directory = "Results/SQLWithSelfexplanationAndSchemalinking"
    spider_dataset = pd.read_csv("spider/Spider_revised.csv",index_col=False)
    spider_dataset = spider_dataset.head(100)
    first_directory_list = load_results(first_directory)
    second_directory_list = load_results(second_directory)
    results = []
    for first,second,NLQ in tqdm(zip(first_directory_list,second_directory_list,spider_dataset.values.tolist()),total=len(first_directory_list)):
        first_acc = get_accuracy(first[0],first[1],first[2])
        second_acc = get_accuracy(second[0], second[1], second[2])
        if first_acc == second_acc and first_acc == float(1):
            decision = "BOTH-CORRECT"
        elif first_acc == second_acc and first_acc == float(0):
            decision = "BOTH-WRONG"
        elif first_acc != second_acc and first_acc == float(0):
            decision = "With Explanation is Correct"
        else:
            decision = "Without Explanation is Correct"
        results.append([first[0],NLQ[2],first[1],first[2],second[2],decision])
    df = pd.DataFrame(results, columns=['DATABASE','QUESTION', 'GOLD SQL', 'SQLAgent SQL', 'SQLDBChainFewShot SQL','DECISION'])
    df.to_csv("analysis_of_explanation.csv", index=False)