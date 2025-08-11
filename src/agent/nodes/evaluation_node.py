import evaluate


def facutality_score():
    pass


def context_recall_score():
    pass


def compute_metrics(expanded_query: str, response: str, ground_truth: str):
    metrics = {}

    squad_metric = evaluate.load("squad")
    bleu = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")

    squad_results = squad_metric.compute(
        predictions=[{"id": "1", "prediction_text": response}],
        references=[{"id": "1", "answers": {"text": [ground_truth], "answer_start": [0]}}],
    )
    metrics["F1"] = squad_results["f1"] / 100.0

    blue_result = bleu.compute(predictions=[response], references=[[ground_truth]])
    blue_result = blue_result["score"] / 100.0
    blue_result = max(0.0, min(1.0, blue_result))
    metrics["Bleu"] = blue_result

    rouge_scores = rouge_metric.compute(
        predictions=[response], references=[ground_truth], rouge_types=["rouge1", "rouge2"]
    )
    metrics["Rouge 1"] = rouge_scores["rouge1"]
    metrics["Rouge 2"] = rouge_scores["rouge2"]
