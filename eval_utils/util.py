import json

def prepare_corpus(annotations):
    return {
        '-'.join((anno['task'], str(anno['conversation'][0]['value']))): [str(anno['conversation'][1]['value'])]
        for anno in annotations
    }

def load_all(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    datasets = ['vqa', 'vln', 'eqa', 'mr', 'bc', 'dc']
    all_corpuses, all_candidates = {}, {}

    for dataset in datasets:
        corpus, candidates = {}, {}
        for key, value in data.items():
            if key.startswith(f'{dataset}-'):
                corpus[key] = value['gt']
                candidates[key] = value['pred']
        
        all_corpuses[dataset.upper()] = (corpus, candidates)
        
    return all_corpuses, all_candidates
