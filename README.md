## An Improved, Strong Baseline for Pre-Trained Large Language Models as Task-Oriented Dialogue Systems 

This is the code for the Findings of the EMNLP 2025 paper: [An Improved, Strong Baseline for Pre-Trained Large Language Models as Task-Oriented Dialogue Systems](https://aclanthology.org/2025.findings-emnlp.605/)

### Installation

1) create new venv environment 
2) activate venv with: source .venv/bin/activate
3) Install requirements:
Use your Conda/Virtualenv and simply run:
```
pip install -r requirements.txt
```


### Usage

4) You need to create the FAISS vector store DB:
```
python create_faiss_db.py --output_faiss_db mwoz_db.pkl --total 20 --dataset multiwoz
```

5) Run prediction for example:
```
python3 run.py --model_name meta-llama/Llama-3.1-8B-Instruct --faiss_db mwoz_db.pkl --num_examples 2 --database_path multiwoz_database --context_size 2 --output out_file.txt --dataset multiwoz --check_domain --check_state --check_response --dials_total=1000 --note "With Self-checking"
```
Or
```
python3 run.py --model_name meta-llama/Llama-3.1-8B-Instruct --faiss_db mwoz_db.pkl --num_examples 2 --database_path multiwoz_database --context_size 2 --output out_file.txt --dataset multiwoz  --dials_total=1000 --note "Without Self-checking"
```
6) Once the script is complete, you will be able to check the results in wandb.

### Acknowledgement

This code builds upon the publication [Are Large Language Models All You Need for Task-Oriented Dialogue?](https://aclanthology.org/2023.sigdial-1.21/) (Hudeček & Dusek, SIGDIAL 2023) and the code published alongside it: [github.com/vojtsek/to-llm-bot](https://github.com/vojtsek/to-llm-bot) 
