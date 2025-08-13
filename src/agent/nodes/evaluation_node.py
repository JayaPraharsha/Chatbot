import evaluate

from agent.db.access import get_answer
from agent.llms import context_recall_evaluator_llm, factuality_evalutator_llm
from agent.prompts.evaluation import context_recall_evaluation_prompt, factuality_evaluation_prompt
from agent.schemas import ContextRecallScore, FactualityScore


def facutality_score(question: str, ground_truth: str, ai_response: str):
    fe_llm = factuality_evalutator_llm.with_structured_output(FactualityScore)
    fe_chain = factuality_evaluation_prompt | fe_llm
    result = fe_chain.invoke(
        {"question": question, "ground_truth": ground_truth, "ai_response": ai_response}
    )
    return result.factuality_score


def context_recall_score(question: str, ground_truth: str, context: str):
    cr_llm = context_recall_evaluator_llm.with_structured_output(ContextRecallScore)
    cr_chain = context_recall_evaluation_prompt | cr_llm
    result = cr_chain.invoke(
        {"question": question, "ground_truth": ground_truth, "context": context}
    )
    return result.context_recall_score


def compute_metrics(
    metric: str, expanded_query: str, initial_user_query: str, response: str, extracted_documents
):
    ground_truth = get_answer(initial_user_query)
    match metric:
        case "F1":
            squad_metric = evaluate.load("squad")
            squad_results = squad_metric.compute(
                predictions=[{"id": "1", "prediction_text": response}],
                references=[{"id": "1", "answers": {"text": [ground_truth], "answer_start": [0]}}],
            )
            return {"F1": squad_results["f1"] / 100.0}

        case "Bleu":
            bleu = evaluate.load("sacrebleu")
            blue_result = bleu.compute(predictions=[response], references=[[ground_truth]])
            blue_result = blue_result["score"] / 100.0
            blue_result = max(0.0, min(1.0, blue_result))
            return {"Bleu": blue_result}

        case "Rouge":
            rouge_metric = evaluate.load("rouge")
            rouge_scores = rouge_metric.compute(
                predictions=[response], references=[ground_truth], rouge_types=["rouge1", "rouge2"]
            )
            return {
                "Rouge 1": rouge_scores["rouge1"].item(),
                "Rouge 2": rouge_scores["rouge2"].item(),
            }

        case "Factuality":
            return {"Factuality": facutality_score(expanded_query, ground_truth, response)}

        case "Context Recall":
            context = " "
            for i, doc in enumerate(extracted_documents):
                context += f"Document {i + 1}:\n{doc}\n\n"
            return {"Context Recall": context_recall_score(expanded_query, ground_truth, context)}
