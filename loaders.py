from datasets import load_dataset
from collections import defaultdict
from typing import Dict, List
from database import MultiWOZDatabase
import copy

def load_mwoz(database_path, context_size, split='train', total=10, skipFirstN=0, shuffle=True, available_domains=None, only_single_domain=False, restrict_domains=None):
    database = MultiWOZDatabase(database_path)
    dataset = load_dataset('multi_woz_v22', trust_remote_code=True)
    if available_domains is not None:
        domain_counts = {d: 0 for d in available_domains}
    else:
        domain_counts = defaultdict(int)
        domain_counts['aux'] = -1
    if shuffle:
        data = dataset[split].shuffle()
    else:
        data = dataset[split]
    n = 1
    slots_per_domain = defaultdict(set)
    domain_counter = defaultdict(int)
    for dialog in data:
        if n <= skipFirstN:
            n += 1
            continue
        if only_single_domain and len(dialog['services']) != 1:
            continue
        if all((dc >= total for dc in domain_counts.values())) or (available_domains is None and n > total):
            break
        dialogue_id = dialog['dialogue_id'].split('.')[0].lower()
        if len(dialog['services']) > 0:
            domain_gt = dialog['services'][0]
        else:
            domain_gt = ''
        for dom in dialog['services']:
            domain_counter[dom] += 1
        if restrict_domains is not None and not all((dom in restrict_domains for dom in dialog['services'])):
            continue
        
        if available_domains and domain_gt in available_domains:
            if domain_counts[domain_gt] >= total:
                continue
            domain_counts[domain_gt] += 1
        n += 1
        last_state = {}
        
        last_domain = ""
        # start iterating over turns within dialog
        for tn in range(0, len(dialog['turns']['utterance']), 2):
            context = [f"Customer: {t}" if n2 % 2 == 0 else f"Assistant: {t}"
             for n2, t in enumerate(dialog['turns']['utterance'][:tn+1])]
            state = dialog['turns']['frames'][tn]['state']
            if len(state) == 0:
                state = {}
            else:
                state = state[0]['slots_values']
                state = {k: v[0] for k, v in zip(state['slots_values_name'], state['slots_values_list']) }
            new_state = {}
            for sl, val in state.items():
                domain, name = sl.split('-')
                slots_per_domain[domain].add(name)
                if domain not in new_state:
                    new_state[domain] = {name: val}
                else:
                    new_state[domain][name] = val
            state_update = {}
            for domain, domain_state in new_state.items():
                for slot, value in domain_state.items():
                    if slot not in last_state.get(domain, {}) or last_state[domain][slot] != value:
                        if domain not in state_update:
                            state_update[domain] = {}
                        state_update[domain][slot] = value
            last_state = new_state
            database_results = {domain: len(database.query(domain, domain_state))
                                for domain, domain_state in new_state.items()}
            
            # get the service = domain for the current turn
            if len(dialog['services']) > 1:
                try:
                    use_domain = dialog["turns"]["frames"][tn]["service"][0]
                except:
                    use_domain = last_domain if len(last_domain) > 0 else domain_gt
            else:
                use_domain = domain_gt
            
            last_domain = copy.copy(use_domain)
            turn = {'page_content': '\n'.join(context[-context_size:]),
                    'question': dialog['turns']['utterance'][tn],
                    'gt_state': last_state,
                    'dialogue_id': dialogue_id,
                    'metadata': {'domain': use_domain,
                                 'state': state_update,
                                 'full_state': last_state,
                                 'context': '\n'.join(context[-6:]),
                                 'response': delexicalize_mwoz(dialog['turns']['utterance'][tn+1],
                                                               dialog['turns']['dialogue_acts'][tn+1]['span_info']),
                                 'database': database_results}}
            yield turn
    print(slots_per_domain)
    print(domain_counter)


def delexicalize_mwoz(utterance: str, span_info: Dict[str, List[str]]):
    for s_idx in range(len(span_info['act_slot_name']) - 1, -1, -1):
        name = span_info['act_slot_name'][s_idx]
        placeholder = f'[{name}]'
        utterance = utterance[:span_info['span_start'][s_idx]] + placeholder + utterance[span_info['span_end'][s_idx]:]
    return utterance