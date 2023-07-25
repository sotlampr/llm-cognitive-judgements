#!/usr/bin/env python3
import csv
import json
from operator import itemgetter
import urllib.request

from utils import generate_prompts

extra_prompt = " The answer must be a single number, not a range, with no explanation."

with open(".openai-api-key") as fp:
    api_key = fp.read().strip()


def query_gpt3(prompts):
    req = urllib.request.Request(
        "https://api.openai.com/v1/completions",
        data=json.dumps(dict(
            model="text-davinci-003",
            prompt=prompts
        )).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    return list(
        map(
            str.strip,
            map(
                itemgetter("text"),
                sorted(json.load(res.fp)["choices"], key=itemgetter("index"))
            )
        )
    )


def query_chatgpt(prompt, model):
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(dict(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    return json.load(res.fp)["choices"][0]["message"]["content"].strip()


def query_gpt35(prompt):
    return query_chatgpt(prompt, "gpt-3.5-turbo")


def query_gpt4(prompt):
    return query_chatgpt(prompt, "gpt-4")


def make_batches(iterable, n=1):
    tmp = list()
    i = 1
    for x in iterable:
        tmp.append(x)
        if i % n == 0:
            yield tmp
            tmp = list()
        i += 1
    if tmp:
        yield tmp


print("Quering GPT3", end="", flush=True)
with open("results/gpt3.csv", "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(("task", "value", "response"))
    for batch in make_batches(generate_prompts(extra_prompt=extra_prompt), 20):
        tasks, values, prompts = zip(*batch)
        prompts = [x+"Answer: " for x in prompts]
        responses = query_gpt3(prompts)
        writer.writerows(zip(tasks, values, responses))
        print(".", end="", flush=True)
print("\tDONE")

print("Quering GPT3.5", end="", flush=True)
try:
    raise FileNotFoundError
    with open("results/gpt35.csv", "r", newline="") as fp:
        reader = csv.reader(fp)
        num_rows = len(fp.readlines()) - 1
except FileNotFoundError:
    num_rows = 1
    with open("results/gpt35.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(("task", "value", "response"))
with open("results/gpt35.csv", "a", newline="") as fp:
    writer = csv.writer(fp)
    for i, (task, value, prompt) in enumerate(
        generate_prompts(extra_prompt=extra_prompt), 1
    ):
        if task != "movie_grosses" or value != 6:
            continue
        if i < num_rows:
            continue
        response = query_gpt35(prompt)
        writer.writerow((task, value, response))
        print(".", end="", flush=True)
        # time.sleep(21)
print("\tDONE")

# print("Quering GPT4", end="", flush=True)
# # with open(f"results/gpt35.csv", "r", newline="") as fp:
# #     reader = csv.reader(fp)
# #     num_rows = len(fp.readlines()) - 1
# # with open("results/gpt4.csv", "a", newline="") as fp:
# with open("results/gpt4.csv", "w", newline="") as fp:
#     writer = csv.writer(fp)
#     writer.writerow(("task", "value", "response"))
#     for i, (task, value, prompt) in enumerate(generate_prompts(extra_prompt=extra_prompt), 1):
#         # if i <= num_rows:
#         #     continue
#         response = query_gpt4(prompt)
#         writer.writerow((task, value, response))
#         print(".", end="", flush=True)
#         time.sleep(21)
# print("\tDONE")
