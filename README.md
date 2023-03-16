# Few-shot-NL2SQL-with-prompting

## Table of contents
* [Dataset](#dataset)
* [Setup](#setup)


## dataset
To reproduce the results reported on the paper please donwload the Spider dataset from the link bellow, and create a data directory containing the tables.json and dev.json.

```
$ Spider dataset = "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"
```


## setup
To run this project, use the following commands:

```
$ pip3 install -r requirements.txt
$ echo "Start running test.py"
$ python3 CoT.py --dataset ./data/ --output predicted_sql.txt
$ echo "Finished running test.py"
```

