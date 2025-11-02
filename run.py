import argparse
import pickle
import json
import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from pynvml import *
from datasets import load_dataset
import wandb
import logging
import transformers
import random
import re
import copy
from model import FewShotPromptedLLM, SimplePromptedLLM
from loaders import load_mwoz
from delex import prepareSlotValuesIndependent, delexicalise, delexicaliseReferenceNumber
from definitions import (MW_FEW_SHOT_DOMAIN_DEFINITIONS, MW_ZERO_SHOT_DOMAIN_DEFINITIONS, MWZ_SLOT_PER_DOMAIN, 
                         multiwoz_domain_prompt, domain_prompt_check_response, response_prompt_check_response, 
                         state_prompt_check_response)
from database import MultiWOZDatabase
from utils import (parse_state, ExampleRetriever, ExampleFormatter, rename_or_create_file,
                   getStateTemplate, set_all_seeds)
from mwzeval.metrics import Evaluator as MWEvaluator
import mwzeval.metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

set_all_seeds(42)

def check(check_model, check_tokenizer, prompt, prev_in, prev_out):
    """
    Check the model's response to a given prompt.
    Args:
        check_model: The model to use for checking.
        check_tokenizer: The tokenizer to use for the model.
        prompt: The prompt to check.
        prev_in: The previous input to the model (for context).
        prev_out: The previous output from the model (for context).

    Returns:
        The model's response to the prompt.
    """
    global APPLY_CHAT_TEMPLATE
    global IS_SEQ2SEQ
    messages = [    
            {  "role": "system",
                            "content": "You are an assistant who always responds exactly like defined by the user. You never respond with code, but instead solve the task directly. Do not respond with more than the user asked for. Give the answer and only the answer directly. Do Not use starting phrases like 'Note:', 'topic:', 'state:', 'response:' or similiar. Do not explain your answer or reasoning. Do not repeat examples from the instruction or add your own examples.", },
                        {"role": "user", "content": prev_in},
                {"role": "assistant", "content": prev_out},
                {"role": "user", "content": prompt},
                     ]
    if APPLY_CHAT_TEMPLATE:
        inputs= check_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt", return_dict=True)
        input_ids = inputs["input_ids"].to(check_model.device)
    else:
        message_list = []
        for m in messages:
            message_list.append(f"{m['role']}: {m['content']}")
        
        inputs= check_tokenizer("\n".join(message_list), return_tensors="pt")
        input_ids = inputs["input_ids"].to(check_model.device)
   
    
    max_length = max_new_tokens = 80
    max_length = input_ids.shape[1] + max_length
    output = check_model.generate(input_ids,
                                     do_sample=True,
                                     top_p=0.9,
                                     pad_token_id=check_tokenizer.eos_token_id,
                                     max_new_tokens=max_new_tokens,
                                     temperature=0.01, attention_mask=inputs["attention_mask"])
    if not IS_SEQ2SEQ:
        output = output[0, input_ids.shape[1]:]
    else: 
        output = output[0]
    output = check_tokenizer.decode(output, skip_special_tokens=True)
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./hf_cache")
    parser.add_argument("--model_name", type=str, default="allenai/tk-instruct-3b-def-pos-neg-expl")
    parser.add_argument("--faiss_db", type=str, default="mwoz_db.pkl")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--dials_total", type=int, default=100)
    parser.add_argument("--skipFirstN", type=int, default=0)
    parser.add_argument("--database_path", type=str, default="multiwoz_database")
    parser.add_argument("--dataset", type=str, default="multiwoz")
    parser.add_argument("--context_size", type=int, default=2)
    parser.add_argument("--ontology", type=str, default="ontology.json")
    parser.add_argument("--output", type=str, default="results.txt")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--use_gt_state", action='store_true')
    parser.add_argument("--use_gt_domain", action='store_true')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--use_zero_shot", action='store_true')
    parser.add_argument("--from_wandb_id", type=str)
    parser.add_argument("--single_domain", action='store_true')
    parser.add_argument("--restrict_domains", type=str)
    parser.add_argument("--notes", type=str, help="Custom note for the W&B run")
    parser.add_argument("--check_domain", action='store_true')
    parser.add_argument("--check_state", action='store_true')
    parser.add_argument("--check_response", action='store_true')
    parser.add_argument("--eval_log_fname", type=str, default="my_eval_ErrorLog.txt")

    APPLY_CHAT_TEMPLATE = True
    IS_SEQ2SEQ = False    
    args = parser.parse_args()
    rename_or_create_file(args.eval_log_fname)
    mwzeval.metrics.EVAL_ERROR_LOG_FNAME = args.eval_log_fname
    config = {
        "model_name": args.model_name,
        "faiss_db": args.faiss_db,
        "num_examples": args.num_examples,
        "dataset": args.dataset,
        "context_size": args.context_size,
        "use_gt_state": args.use_gt_state,
        "use_zero_shot": args.use_zero_shot,
        "use_gt_domain": args.use_gt_domain,
        "split": args.split,
        "num_dialogs": args.dials_total,
        "skipFirstN": args.skipFirstN,
        "check_domain": args.check_domain,
        "check_state": args.check_state,
        "check_response": args.check_response,
    }


    if 'tk-instruct-3b' in args.model_name:
        model_name = 'tk-3B'
    elif 'tk-instruct-11b' in args.model_name:
        model_name = 'tk-11B'
    elif 'opt-iml-1.3b' in args.model_name:
        model_name = 'opt-iml-1.3b'
    elif 'opt-iml-30b' in args.model_name:
        model_name = 'opt-iml-30b'
    elif 'NeoXT' in args.model_name:
        model_name = 'togethercomputer/GPT-NeoXT-Chat-Base-20B'
    else:
        model_name = args.model_name


    if args.from_wandb_id is None:
        # Wandb Initialization
        wandb.init(project='llmbot', config=config, notes=args.notes)
        run_name_gtState = "-gtState" if args.use_gt_state else ""
        RUN_NAME =  f'{args.run_name}-{args.dataset}-{model_name}-examples-{args.num_examples}-ctx-{args.context_size}{run_name_gtState}'
        wandb.run.name = RUN_NAME
        report_table = wandb.Table(columns=['id', 'context', 'raw_state', "turn_state", 'total_state', "gt_state", 'response', "gt_response", 'predicted_domain', "gt_domain", 'domain prompt', "state prompt", "response prompt", "database_result"])
        wandb_run = None

        if any([n.lower() in args.model_name.lower() for n in ['opt', 'NeoXT', "llama", "mixtral", "phi", "qwen"]]):
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
            model_w = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                           low_cpu_mem_usage=True,
                                                           cache_dir=args.cache_dir,
                                                           device_map="auto",
                                                           load_in_8bit=True)
            model_factory = SimplePromptedLLM if args.use_zero_shot else FewShotPromptedLLM
            model = model_factory(model_w, tokenizer, type="causal")
            domain_model = SimplePromptedLLM(model_w, tokenizer, type="causal")
            if any([n.lower() in args.model_name.lower() for n in ['opt', "neoxt"]]):
                APPLY_CHAT_TEMPLATE = False
        else:
            # TK Instruct
            APPLY_CHAT_TEMPLATE = False
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
            model_w = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                        low_cpu_mem_usage=True,
                                                        cache_dir=args.cache_dir,
                                                        device_map="auto",
                                                        load_in_8bit=True)
            model_factory = SimplePromptedLLM if args.use_zero_shot else FewShotPromptedLLM
            model = model_factory(model_w, tokenizer, type="seq2seq")
            domain_model = SimplePromptedLLM(model_w, tokenizer, type="seq2seq")

        # Setup model for checking
        check_model = model_w
        check_tokenizer = tokenizer
    else:
        wandb_run = wandb.init(project="llmbot", id=args.from_wandb_id)
    
    # Load FAISS DB, Ontology and Prepare Example Retriever and Formatter
    with open(args.faiss_db, 'rb') as f:
        faiss_vs = pickle.load(f)
    with open(args.ontology, 'r') as f:
        ontology = json.load(f)
    if args.dataset == 'multiwoz':
        domain_prompt = multiwoz_domain_prompt
        database = MultiWOZDatabase(args.database_path)
        state_vs = faiss_vs
        delex_dic = prepareSlotValuesIndependent(args.database_path)
    example_retriever = ExampleRetriever(faiss_vs)
    state_retriever = ExampleRetriever(state_vs)
    example_formatter = ExampleFormatter(ontology=ontology)

    # Load Database
    history = []
    n = 0
    results = {}
    results_wo_state = {}
    last_dial_id = None
    total = args.dials_total + args.skipFirstN
    print("TOTAL", total)
    if args.dataset == 'multiwoz':
        data_gen = \
            load_mwoz(args.database_path, args.context_size, split=args.split, total=total, skipFirstN=args.skipFirstN, shuffle=False, only_single_domain=args.single_domain, restrict_domains=args.restrict_domains.split(",") if args.restrict_domains is not None else None)
    progress_bar = tqdm.tqdm(total=args.dials_total)
    predictions_table = None

    # Iterate over the datase
    domain_correct, total_turns = 0, 0
    for it, turn in enumerate(data_gen):
        total_turns += 1
        if last_dial_id != turn['dialogue_id']:
            last_dial_id = turn['dialogue_id']
            n += 1
            progress_bar.update(1)
            tn = 0
            if n > total:
                break
            history = []
            dialogue_id = turn['dialogue_id']
            results[dialogue_id] = []
            results_wo_state[dialogue_id] = []
            # new dialog, so start with a new aggregated gt state and aggregated predicted state(=total_state)
            aggregated_gt_state = {}
            total_state = {}
            if n < 5:
                print('=' * 100)
            previous_domain = None
        if False:
            pass
        else:
            question = turn['question']
            gold_response = turn['metadata']['response']
            gt_state = turn['gt_state']
            gt_domain = turn['metadata']['domain']
            if len(gt_state) == 0:
                gt_state = {}
            
            for domain, ds in gt_state.items():
                for sl, val in ds.items():
                    if domain not in aggregated_gt_state:
                        aggregated_gt_state[domain] = {sl: val}
                    else:
                        aggregated_gt_state[domain][sl] = val
            retrieve_history = history + ["Customer: " + question]
            retrieved_examples = example_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]), k=20)
            retrieved_domains = [example['domain'] for example in retrieved_examples]
            
            if args.dataset == 'multiwoz':
                available_domains = list(map(str.lower, MW_FEW_SHOT_DOMAIN_DEFINITIONS.keys()))

            # Predict Domain
            domainIsValid = False 
            domainTryCounter = 1
            didUseRandomDomain = False
            while domainTryCounter >= 0: # try domainTryCounter times to find a domain within the list of available domains
                domainTryCounter -= 1
                if not domainIsValid:
                    selected_domain, dp = domain_model(domain_prompt, predict=True, history="\n".join(history), utterance=f"Customer: {question.strip()}")
                    prev_selected_domain = selected_domain
                    if "\n" in selected_domain:
                        selected_domain = selected_domain.split("\n", 1)[0]
                        
                    selected_domain = re.sub(r'[^a-z]', '', selected_domain.lower()) # remove everything that isnt a-z
                    
                    if selected_domain in available_domains: # found valid, skip rest
                        domainIsValid = True 
                        domainTryCounter = -1
                    else:
                        if args.check_domain:
                            selected_domain_parsed = check(check_model, check_tokenizer, domain_prompt_check_response, prev_in=dp,
                                                        prev_out=prev_selected_domain)
                            selected_domain_parsed = re.sub(r'[^a-z]', '', selected_domain_parsed.lower())   # remove everything that isnt a-z
                            if n < 5:
                                print(f"SENTENCE {question}, FIRST DOMAIN: {selected_domain}, PARSED DOMAIN: {selected_domain_parsed}," \
                                   f"GROUND_TRUTH: {gt_domain}")
                            selected_domain = selected_domain_parsed
                            
                            if selected_domain in available_domains: # found valid, skip rest
                                domainIsValid = True 
                                domainTryCounter = -1
                    if n < 5: 
                       print(f"PREDICTED DOMAIN: {selected_domain}, GROUND_TRUTH: {gt_domain}, COUNTER: {domainTryCounter}")
                    
            
            # no valid found, choose random
            if selected_domain not in available_domains:
                temp_selected_domain = copy.copy(selected_domain)
                for dom in available_domains:
                    if dom in selected_domain:
                        temp_selected_domain = dom
                if temp_selected_domain == selected_domain: # we haven't found a domain from the list in the string 
                    # so use random
                    selected_domain = random.choice(available_domains)
                    didUseRandomDomain = True
                else:
                    selected_domain = copy.copy(temp_selected_domain)
                del temp_selected_domain
                
            if args.dataset == 'multiwoz':
                domain_definition = MW_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain] if args.use_zero_shot else MW_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
            most_common_domain = Counter(retrieved_domains).most_common(1)[0][0]
            
            if selected_domain.lower() == gt_domain.lower():
                domain_correct += 1
            
            if args.use_gt_domain:
                selected_domain = gt_domain
                                       
            if previous_domain != selected_domain:
                previous_domain = selected_domain
            retrieved_examples = [example for example in retrieved_examples if example['domain'] == selected_domain]
            num_examples = min(len(retrieved_examples), args.num_examples)
            num_state_examples = args.num_examples
            state_examples = [example for example in state_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]), k=20) if example['domain'] == selected_domain][:num_state_examples]
            positive_state_examples = example_formatter.format(state_examples[:num_state_examples],
                                                               input_keys=["context"],
                                                               output_keys=["state"],
                                                               use_json=False,
                                                               domain=selected_domain
                                                               )
            
            response_examples = example_formatter.format(retrieved_examples[:num_examples],
                                                         input_keys=["context", "full_state", "database"],
                                                         output_keys=["response"],
                                                         use_json=False,
                                                         domain=selected_domain)
            
            state_prompt = domain_definition.state_prompt
            response_prompt = domain_definition.response_prompt
            

            # Predict State
            if args.use_gt_state:
                state = str(aggregated_gt_state)
                parsed_state = total_state = final_state = aggregated_gt_state
                filled_state_prompt = "USED GT_STATE"
            else:
                try:
                    state_template =  getStateTemplate(total_state, selected_domain)
                    kwargs = {
                        "history": "\n".join(history),
                        "utterance": question.strip(),
                        "state": state_template
                    }
                    if not args.use_zero_shot:
                        kwargs["positive_examples"] = positive_state_examples
                        kwargs["negative_examples"] = [] # negative_state_examples
                    state, filled_state_prompt = model(state_prompt, predict=True, **kwargs)
                    state = state.strip()
                    
                    if args.check_state:
                        slotList = MWZ_SLOT_PER_DOMAIN[selected_domain]
                        state_parsed = check(check_model, check_tokenizer, state_prompt_check_response.substitute(state=state_template), prev_in=filled_state_prompt, prev_out=state)
                        state_parsed = state_parsed.strip()
                        if n < 5:
                            print(f"FIRST STATE RESPONSE: {state}, PARSED STATE RESPONSE: {state_parsed}")
    
                        state = state_parsed          
                    
                    if n < 2:
                        print("Filled prompt:", filled_state_prompt)
                except Exception as e:
                    state = "{}"
                    filled_state_prompt = "EXCEPTION, USING EMPTY STATE"

                # parse state
                parsed_state = parse_state(state, default_domain=selected_domain)
                if selected_domain not in parsed_state:
                    parsed_state[selected_domain] = {}
                    
                if not isinstance(parsed_state[selected_domain], dict):
                    parsed_state[selected_domain] = {}
                keys_to_remove = [k for k in parsed_state[selected_domain].keys() if k not in domain_definition.expected_slots]
                try:
                    for domain, ds in parsed_state.items():
                        for slot, value in ds.items():
                            pass
                except:
                    parsed_state = {selected_domain: {}}
                
                final_state = {}
                for domain, ds in parsed_state.items():
                    if domain in available_domains:
                        final_state[domain] = ds
                
                for domain, dbs in final_state.items():
                    if domain not in total_state:
                        total_state[domain] = dbs
                    else:
                        for slot, value in dbs.items():
                            value = str(value)
                            total_state[domain][slot] = value
            if n < 5:
                print('-' * 100)
                print(f"Question: {question}", flush=True)
                print(f"Selected domain: {selected_domain}", flush=True)
                logger.info(f"Raw State: {state}")
                print(f"Raw State: {state}", flush=True)
                logger.info(f"Parsed State: {final_state}")
                print(f"Parsed State: {final_state}", flush=True)
                logger.info(f"Total State: {total_state}")
                print(f"Total State: {total_state}", flush=True)

            if args.dataset == 'multiwoz':
                database_results = {domain: len(database.query(domain=domain, constraints=ds))
                                    for domain, ds in total_state.items() if len(ds) > 0}
            else:
                database_results = turn['metadata']['database']
            logger.info(f"Database Results: {database_results}")
            if n < 5:
                print(f"database Results: {database_results}", flush=True)
            
            def getDBString(db):
                db_string = ""
                for domain, value in db.items():
                    numerator = "" if value <= 1 else "s"
                    db_string += f"The database returned {value} hit{numerator} for the {domain}.\n"
                return db_string
            
            try:
                kwargs = {
                    "history": "\n".join(history),
                    "utterance": question.strip(),
                    "state": getStateTemplate(total_state, selected_domain),
                    "database": getDBString(database_results)
                }
                if not args.use_zero_shot:
                    kwargs["positive_examples"] = response_examples
                    kwargs["negative_examples"] = []

                # Predict Response
                response, filled_prompt = model(response_prompt, predict=True, **kwargs)
                response = response.strip()
                if args.check_response:
                    response_checked = check(check_model, check_tokenizer, response_prompt_check_response, prev_in=filled_prompt, prev_out=response)
                    response_checked = response_checked.strip()
                    if n < 5:
                        print(f"FIRST RESPONSE: {response}, CHECKED RESPONSE: {response_checked}")
                    response = response_checked
                if n < 2:
                    print("Filled response prompt:", filled_prompt)
            except:
                response = ''

            if args.dataset == 'multiwoz':
                response = delexicalise(response, delex_dic)
                response = delexicaliseReferenceNumber(response)
            
            if n < 5:
                print(f"Response: {response}", flush=True)
                print(f"Gold Response: {gold_response}", flush=True)

            history.append("Customer: " + question)
            report_table.add_data(f"{dialogue_id}-{tn}", " ".join(history), state, str(final_state), str(total_state), str(aggregated_gt_state), response, gold_response, selected_domain, gt_domain, dp, filled_state_prompt, filled_prompt, str(database_results))
            history.append("Assistant: " + gold_response)
            
            results[dialogue_id].append({
                "domain": copy.deepcopy(selected_domain),
                "active_domains": [copy.deepcopy(selected_domain)],
                "response": copy.deepcopy(response),
                "state": copy.deepcopy(total_state),
            })
            results_wo_state[dialogue_id].append({
                "domain": copy.deepcopy(selected_domain),
                "active_domains": [copy.deepcopy(selected_domain)],
                "response": copy.deepcopy(response),
            })
        if n == 5 and tn == 1:
            wandb.log({"preview_examples": report_table})
            
        tn += 1
    progress_bar.close()
    
    # wandb: new object otherwise, preview get's overwritten
    final_report_table = wandb.Table(columns=['id', 'context', 'raw_state', "turn_state", 'total_state', "gt_state", 'response', "gt_response", 'predicted_domain', "gt_domain", 'domain prompt', "state prompt", "response prompt", "database_result"], data=report_table.data)
    wandb.log({"final_predictions": final_report_table}) 

    rename_or_create_file(args.output)
    with open(args.output, "w") as f:
        f.write(f"Results: {str(config)}")
    
    # Evaluate
    if args.dataset == 'multiwoz':
        evaluator = MWEvaluator(bleu=True, success=True, richness=True, jga=True, dst=True)
        eval_results = evaluator.evaluate(results)
        print(eval_results)
        for metric, values in eval_results.items():
            if values is not None:
                for k, v in values.items():
                    wandb.log({f"MW_{metric}-{k.ljust(15)}": v})
                    with open(args.output, "a") as f:
                        f.write("\n" + str({f"MW_{metric}-{k.ljust(15)}": v}))

        evaluator = MWEvaluator(bleu=True, success=True, richness=True)
        eval_results = evaluator.evaluate(results_wo_state)
        for metric, values in eval_results.items():
            if values is not None:
                for k, v in values.items():
                    wandb.log({f"MW_GT_{k.ljust(15)}": v})
                    with open(args.output, "a") as f:
                        f.write("\n" + str({f"MW_{metric}-{k.ljust(15)}": v}))

    wandb.log({"percentDomainCorrect": domain_correct/total_turns})
  
    eval_log_artifact = wandb.Artifact(name="Eval-log", type="eval-log")
    eval_log_artifact.add_file(local_path=args.eval_log_fname)
    wandb.log_artifact(eval_log_artifact)
