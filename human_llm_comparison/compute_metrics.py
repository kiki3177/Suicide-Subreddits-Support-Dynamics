from bert_score import score as bert_score_fn
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')


class RewardFunction:
    def __init__(self):
        self.rouge_scorer = evaluate.load('rouge')

    def compute_reward(self, generated: str, reference: str, metric: str = "bertscore") -> float:
        """
        Evaluation using the specified metric: 'rouge_l', 'bertscore', or 'bleu_score'.
        """
        if metric not in ["rouge_l", "bertscore", "bleu_score"]:
            raise ValueError(f"Unsupported metric: {metric}")

        if metric == "rouge_l":
            score  = self.rouge_scorer.compute(predictions=[generated], references=[reference])["rougeL"]
        elif metric == "bertscore":
            P, R, F1 = bert_score_fn([generated], [reference], lang='en', verbose=False)
            score = F1.item()
        elif metric == "bleu_score":
            reference_tokens = word_tokenize(reference.lower())
            generated_tokens = word_tokenize(generated.lower())
            smoothie = SmoothingFunction().method4
            score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothie)
        return score
