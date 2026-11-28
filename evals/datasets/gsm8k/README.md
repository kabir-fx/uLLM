# GSM8K, first 200 test items

The exact-match task set of Q5 (decision 0035).

- **Source:** GSM8K, Cobbe et al., "Training Verifiers to Solve Math Word Problems" (2021), <https://github.com/openai/grade-school-math>.
- **File:** `grade_school_math/data/test.jsonl` at commit `3101c7d5072418e28b9008a6636bde82a006892c`, sha256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`, 1,319 lines.
- **This copy:** its first 200 lines, unchanged. Each line is one item, `{"question", "answer"}`, and the gold answer follows `#### ` in `answer`.
- **License:** MIT, copyright (c) 2021 OpenAI. The full text is in [LICENSE](LICENSE).

Made with:

```sh
curl -sSfL https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl | head -n 200 > test-200.jsonl
```

`evals/data.py` refuses a copy whose sha256 is not the one it records.
