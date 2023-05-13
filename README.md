# Few-shot-NL2SQL-with-prompting

## Table of contents
* [Dataset](#dataset)
* [Setup](#setup)
* [Citation](#citation)


## dataset
To reproduce the results reported in the paper, please download the Spider dataset from the link below and create a data directory containing the tables.json and dev.json files.

```
$ Spider dataset = "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"
```


## setup
To run this project, use the following commands:

```
$ pip3 install -r requirements.txt
$ echo "Start running DIN-SQL.py"
$ python3 DIN-SQL.py --dataset ./data/ --output predicted_sql.txt
$ echo "Finished running DIN-SQL.py"
```
## citation 

``` 
@article{pourreza2023din,
  title={DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction},
  author={Pourreza, Mohammadreza and Rafiei, Davood},
  journal={arXiv preprint arXiv:2304.11015},
  year={2023}
}
```

