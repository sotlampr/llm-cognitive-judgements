# LLM Cognitive Judgements...
### ...Differ From Human

Work published as an [arXiv pre-print](https://arxiv.org/abs/2307.11787)

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/)

[![License: CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)


[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2307.11787.svg)]( https://doi.org/10.48550/arXiv.2307.11787)

## Repository Structure

#### `data`
- `data/{TASK}-xaxis.csv`: The $t$ values that were used as prompt from [^1]
- `data/{TASK}-participants.csv`: Participants' answers for $t_total$ from [^1]
- `data/{TASK}-model.csv`: Model predictions for $t_total$ from [^1]

#### `data/prompts`
- `data/prompts/{TASK}`: Prompts for each task with `%d` placeholder for the actual value from [^1]

#### `src`
- `src/utils.py`: Helper utilities
- `src/query-openai.py`: Script to query OpenAI GPT-3 and ChatGPT with the tasks
- `src/plot.py`: Script to reproduce the plots and tables in the paper (needs [texfig](https://github.com/nilsvu/texfig))

#### `results`
- `results/gpt3.csv`: GPT-3 Answers
- `results/gpt35.csv`: ChatGPT Answers

[^1]: Griffiths, Thomas L., and Joshua B. Tenenbaum. "Optimal predictions in everyday cognition." Psychological science 17.9 (2006): 767-773.
