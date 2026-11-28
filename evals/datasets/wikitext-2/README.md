# WikiText-2, raw test text

The fixed text for perplexity in Q5 (decision 0035).

- **Source:** WikiText-2 (raw), Merity, Xiong, Bradbury and Socher, "Pointer Sentinel Mixture Models" (2016), <https://arxiv.org/abs/1609.07843>.
- **File:** `wikitext-2-raw/wiki.test.raw` from <https://wikitext.smerity.com/wikitext-2-raw-v1.zip>, an archive with sha256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.
- **This copy:** that file, unchanged, 1,290,590 bytes.
- **License:** Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0), <https://creativecommons.org/licenses/by-sa/3.0/>. The text comes from Wikipedia articles by their contributors. The dataset is also available under the GNU Free Documentation License. This file alone carries these terms. The rest of the repository stays under its MIT license.

Made with:

```sh
curl -sSfL -o wikitext-2-raw-v1.zip https://wikitext.smerity.com/wikitext-2-raw-v1.zip
unzip -p wikitext-2-raw-v1.zip wikitext-2-raw/wiki.test.raw > wiki.test.raw
```

`evals/data.py` refuses a copy whose sha256 is not the one it records.
