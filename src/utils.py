NUM_RUNS = 20

tasks = ["cakes", "life_spans", "movie_grosses", "poems", "representatives", "waiting_time"]


def generate_prompts(num_repeats=NUM_RUNS, extra_prompt=""):
    for task in tasks:
        with open(f"data/prompts/{task}") as fp:
            prompt = fp.read().strip()
        with open(f"data/{task}-xaxis.csv") as fp:
            values = list(map(int, fp.read().strip().split("\n")))
        for value in values:
            for _ in range(num_repeats):
                yield task, value, (prompt % value) + extra_prompt
