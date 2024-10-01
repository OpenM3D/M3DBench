import time
import torch
from eval_utils.util import prepare_corpus, load_all
from eval_utils.score import score_captions
from utils.logger import log_evaluation_results, save_evaluation_results
from utils.misc import SmoothedValue
from utils.dist import is_primary, barrier, all_gather_dict

@torch.no_grad()
def evaluate(args, curr_epoch, model, dataset_config, dataset_loader, logout=print, curr_train_iter=-1):
    # Prepare ground truth caption labels
    print("preparing corpus...")
    annotations = dataset_loader.dataset.annotations
    corpus, candidates = prepare_corpus(annotations), {}

    tokenizer = dataset_loader.dataset.tokenizer
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    
    model.eval()
    barrier()

    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for curr_iter, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        outputs = model(batch_data_label, is_eval=True)
        outputs = dict(output_ids=outputs["output_ids"])
        
        outputs = all_gather_dict(outputs)
        batch_data_label = all_gather_dict(batch_data_label)

        output_ids = outputs["output_ids"]
        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        quesition_index = batch_data_label['quesition_index'].cpu().tolist()
        
        for idx in range(output_ids.shape[0]):
            anno = annotations[quesition_index[idx]]
            key = '-'.join((anno['task'], str(anno['conversation'][0]['value'])))
            candidates[key] = [' '.join(filter(lambda w: w, answers[idx].split(' ')))]

        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logout(f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; Evaluating on iter: {curr_train_iter}; Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB")
        
        barrier()

    score_per_caption, message, eval_metric = score_captions(corpus, candidates)

    if is_primary():
        log_evaluation_results(message, logout)
        save_evaluation_results(args.checkpoint_dir, corpus, candidates, score_per_caption)
        
        json_path = f"{args.checkpoint_dir}/pred_gt_val.json"
        all_corpuses, all_candidates = load_all(json_path)
        
        for corpus_type, (corpus, candidate) in all_corpuses.items():
            _, corpus_message, _ = score_captions(corpus, candidate)
            log_evaluation_results(f"\n----------------------{corpus_type} Evaluation-----------------------\n", logout)
            log_evaluation_results(corpus_message, logout)

    return eval_metric
