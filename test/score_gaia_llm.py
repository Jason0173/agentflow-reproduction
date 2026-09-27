#!/usr/bin/env python3
"""
GAIA LLM-judge scorer — matches AgentFlow's calculate_score_unified.py logic.
Uses gpt-4.1-mini (200K TPM) instead of gpt-4o (30K TPM) to avoid rate limits.
"""

import json, os, re, sys, time, argparse
import openai

VERIFICATION_PROMPT = """
Given a multiple-choice Question, a Model Response, and its Correct Answer, determine whether the Model's prediction is correct.

The prediction is correct only if it **exactly matches** the correct choice letter (e.g., "A", "B", "C", or "D") after necessary normalization. Follow these instructions carefully:

1. If the Model Response is a number (e.g., "2", "3", etc.), map it to the corresponding option letter based on its order in the Question (e.g., 1 → A, 2 → B, etc.).
2. Ignore irrelevant text, explanations, or format differences. Extract the core predicted answer.
3. Compare the final normalized response with the Correct Answer letter.

Question: {question}
Model response: {response}
Correct answer: {correct_answer}

Response Format:
<analysis>: First extract the mathematical answers, then explain the comparison
<true_false>: Return "True" only for exact matches, otherwise "False"
"""


def judge(client, model, question, response, correct_answer):
    # Extract from <answer> tags if present
    matches = re.findall(r"<answer>(.*?)</answer>", str(response), re.DOTALL)
    if matches:
        response = matches[-1].strip()

    prompt = VERIFICATION_PROMPT.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )

    for attempt in range(10):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()

            # Parse true_false from response
            tf_match = re.search(r'<true_false>\s*:?\s*(true|false)', text, re.IGNORECASE)
            if tf_match:
                return tf_match.group(1).lower() == 'true'

            # Fallback: look for True/False anywhere
            text_lower = text.lower()
            if 'true' in text_lower and 'false' not in text_lower:
                return True
            return False

        except openai.RateLimitError:
            wait = 5
            print(f"[RL {attempt+1}]", end=" ", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"[Error: {e}]", end=" ", flush=True)
            return False

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="test")
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge_model", default="gpt-4.1-mini")
    args = parser.parse_args()

    client = openai.OpenAI()

    models = [args.model] if args.model else ['Qwen3.5-0.8B', 'Qwen3.5-2B', 'Qwen3.5-4B', 'Qwen3.5-9B-Modal']

    print(f"Judge: {args.judge_model}\n")

    for model_name in models:
        data_file = os.path.join(args.test_dir, 'gaia', 'data', 'data.json')
        result_dir = os.path.join(args.test_dir, 'gaia', 'results', model_name)
        if not os.path.exists(result_dir):
            print(f"{model_name}: not found, skipping")
            continue

        with open(data_file) as f:
            raw = json.load(f)
        benchmark = {str(d['pid']): d for d in raw}

        results = {}
        for file in os.listdir(result_dir):
            if file.startswith('output_') and file.endswith('.json'):
                try:
                    with open(os.path.join(result_dir, file)) as f:
                        result = json.load(f)
                    pid = str(result['pid'])
                    if pid in benchmark and 'direct_output' in result:
                        ans = benchmark[pid]['answer']
                        if isinstance(ans, list):
                            ans = ans[0]
                        results[pid] = {
                            'question': benchmark[pid].get('question', benchmark[pid].get('query', '')),
                            'response': result['direct_output'],
                            'correct_answer': str(ans),
                        }
                except Exception:
                    pass

        correct = 0
        total = len(results)
        print(f"Scoring gaia/{model_name}: {total} items")

        for i, pid in enumerate(sorted(results.keys(), key=int)):
            item = results[pid]
            is_correct = judge(client, args.judge_model, item['question'], item['response'], item['correct_answer'])
            if is_correct:
                correct += 1
            if (i + 1) % 20 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] correct: {correct} ({correct/(i+1)*100:.1f}%)")

        acc = round(correct / total * 100, 2) if total > 0 else 0.0
        print(f"  Final: {acc}% ({correct}/{total})\n")

        score_file = os.path.join(result_dir, 'final_scores_direct_output.json')
        with open(score_file, 'w') as f:
            json.dump({'correct': correct, 'total': total, 'accuracy': acc}, f, indent=4)


if __name__ == '__main__':
    main()
