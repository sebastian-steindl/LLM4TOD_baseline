import json
import re
import dirtyjson
import random
from copy import deepcopy
from typing import Dict, Any
from collections import defaultdict

import numpy
from fuzzywuzzy import fuzz
import evaluate
from nltk.tokenize import word_tokenize
from langchain.vectorstores import VectorStore

import os
import string
from datetime import datetime

from definitions import MWZ_SLOTS_FOR_TEMPLATE

def getStateTemplate(ts, d):
    currentStateJson = ts.get(d, {}) # fail-safe: if d doesnt exist, return empty
    allSlots = sorted(MWZ_SLOTS_FOR_TEMPLATE.get(d, [])) 
    state_template = ""
    for s in allSlots:
        if s in currentStateJson.keys():
            v = "'" + currentStateJson[s] + "'"
        else:
            v = "'?'"
        state_template += s + ":" + v + "-"
    state_template = state_template[:-1] # remove last hyphen 
    return state_template


def rename_or_create_file(file_path):
    # Check if the path exists
    if os.path.exists(file_path):
        # Extract the file directory, name, and extension
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        file_name, file_ext = os.path.splitext(base_name)
        
        # Generate a random string
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        
        # Create a new unique file name
        new_file_name = f"{file_name}_{random_string}{file_ext}"
        new_file_path = os.path.join(dir_name, new_file_name)
        
        # Rename the file
        os.rename(file_path, new_file_path)
        print(f"File renamed to: {new_file_path}")
    
    # Create the file if it does not exist and write the current datetime
    with open(file_path, 'w') as file:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Created on: {current_datetime}\n")
    print(f"File created at: {file_path}")

def parse_state(state: str, default_domain: str = None) -> Dict[str, str]:
    def sanitize(dct):
        for key in dct:
            if isinstance(dct[key], dict):
                dct[key] = sanitize(dct[key])
            elif not isinstance(dct[key], str):
                dct[key] = str(dct[key])
        return dct
    state = str(state)
    slotvals = re.findall("([a-z]+:('(([a-z]| |[A-Z]|:|[0-9])+')|[A-Za-z0-9:]+))", state)
    out_state = {}
    for sv in slotvals:
        sv = sv[0].strip("'\"").split(':')
        out_state[sv[0].strip("'\"")] = ":".join(sv[1:]).strip("'\" ")
    return {default_domain: sanitize(out_state)}


class ExampleRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, text: str, k: int = 2) -> list[Dict]:
        result = self.vector_store.similarity_search(text, k=k)
        examples = [{'context': doc.metadata['context'],
                     'state': doc.metadata['state'],
                     'full_state': doc.metadata['full_state'],
                     'response': doc.metadata['response'],
                     'database': doc.metadata['database'],
                     'domain': doc.metadata['domain']}
                     for doc in result]
        return examples



class ExampleFormatter:
    def __init__(self, ontology: Dict):
        self.ontology = ontology

    def format(self,
               examples: list[Dict[str, Any]],
               input_keys: list[str],
               output_keys: list[str],
               use_json: bool = False,
               corrupt_state: bool = False,
               domain: str = "") -> list[Dict[str, str]]:

        examples = deepcopy(examples)
        if corrupt_state:
            examples = [self._corrupt_example(example) for example in examples]
        examples = [self._example_to_str(example, use_json, domain) for example in examples]

        def _prepare_example(example: Dict) -> Dict:
            example['input'] = '\n'.join((f"{key if key != 'full_state' else 'state'}: {example[key]}" for key in input_keys))
            example['output'] = '\n'.join((f"{key}: {example[key]}" for key in output_keys))
            return example
        examples = [_prepare_example(example) for example in examples]

        return examples

    def _corrupt_example(self, example: Dict) -> Dict:
        for domain, dbs in example['state'].items():
            for slot, value in dbs.items():
                slot_otgy_name = f"{domain}-{slot}"
                if slot_otgy_name in self.ontology:
                    example['state'][domain][slot] = random.choice(self.ontology[slot_otgy_name])
                else:
                    otgy_key = random.choice(list(self.ontology.keys()))
                    example['state'][domain][slot] = random.choice(self.ontology[otgy_key])
        return example

    def _example_to_str(self, example: Dict, use_json=False, domain: str = "") -> Dict:
        for key, val in example.items():
            if isinstance(val, dict):
                if use_json:
                    example[key] = json.dumps(val)
                else:
                    if key == "state" or key == "full_state":
                        if len(domain) < 1:
                            example[key] = "-".join((f"{slot}:'{value}'" for slot, value in val.items()))
                        else:
                            example[key] = getStateTemplate(example[key], domain)
                    if key == "database":
                        db_string = ""
                        for domain, value in val.items():
                            numerator = "" if value <= 1 else "s"
                            db_string += f"The database found {value} hit{numerator} for {domain}.\n"
                        example[key] = db_string
            else:
                example[key] = str(val)
        return example


def print_gpu_utilization():
    print("no.")

def set_all_seeds(seed_value=42):
    import random
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(seed_value)
    
    np.random.seed(seed_value)
    
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) 
    
    set_seed(seed_value)