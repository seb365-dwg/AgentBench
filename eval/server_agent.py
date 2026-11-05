# Adapted from https://github.com/THUDM/AgentRL (examples/eval/server_agent.py).
# Local changes: --extra-body (merge JSON into each chat request), --limit N (first N samples,
# works for string indices), resume fix (task indices compared as strings), and a Python 3.9-compatible
# traceback.print_exception call.
import asyncio
import os
import random
import re
import sys
import traceback
from argparse import ArgumentParser
from datetime import datetime
from json import JSONDecodeError
from os.path import dirname, join
from typing import List, Tuple, Union, Optional

import pandas as pd
from aiohttp import ClientSession
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


EXTRA_BODY = {}  # set from --extra-body


class OpenAIAgent:
    def __init__(
        self,
        model: str,
        base_url: str,
        model2: str = "",
        base_url2: str = "",
        api_key: str = "placeholder",
        verbose: bool = False,
    ):
        self.model = model
        self.verbose = verbose
        self.base_url = base_url
        self.api_key = api_key

        self.model2 = model2
        self.base_url2 = base_url2

        # Create async client
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.client_session = ClientSession()

    async def call(self, url, model, messages, tools, temperature):
        payload = {
            "model": model,
            "messages": messages,
            "tool_choice": "auto",
            "tools": tools,
            "temperature": temperature,
            "max_completion_tokens": 1024,
        }
        payload.update(EXTRA_BODY)
        response = await self.client_session.post(url + "/chat/completions", json=payload, headers={
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        })
        try:
            data = await response.json(content_type=None)
        except JSONDecodeError:
            data = await response.text()
            print(f"Error: {data}", flush=True)
            raise
        if response.status != 200:
            print(payload)
            print(f"Error", data, flush=True)
        response.raise_for_status()
        message = data['choices'][0]['message']
        if message["content"] is None:
            message["content"] = ""
        message = {k: v for k, v in message.items() if v is not None}

        return message

    async def chat_completion(self, messages: List[dict], tools: Optional[List[dict]] = None, temperature=None) -> dict:
        max_retries = 3

        for tries in range(max_retries):
            try:
                if self.model2 and random.random() < 0.5:
                    return await self.call(self.base_url2, self.model2, messages, tools, temperature)
                return await self.call(self.base_url, self.model, messages, tools, temperature)
            except Exception:
                if tries == max_retries - 1:
                    raise

                if self.verbose:
                    traceback.print_exc()
                    print(f"API call failed (attempt {tries + 1}/{max_retries})", flush=True)
                await asyncio.sleep(1)  # Brief delay before retry

        raise NotImplementedError("Unreachable code")


async def worker(args, session: ClientSession, agent: OpenAIAgent, task_index: str) -> Tuple[
    List[dict], Union[int, Exception], str]:
    while True:
        try:
            response = await session.post(f'{args.controller}/start_sample', json={
                'name': args.task_name[0],
                'index': task_index
            })
            text = await response.text()
            response.raise_for_status()
            session_id = response.headers['session_id']
            break
        except:
            if args.verbose:
                print(f'Failed to start sample for task {task_index}, {text=}', file=sys.stderr)
                traceback.print_exc()
            await asyncio.sleep(1)
    messages = []
    try:
        data = await response.json()
        messages.extend(data['messages'])
        tools = data['tools']

        if args.verbose:
            print(f'session_id: {session_id}')
            print(f"{messages=} {tools=}", flush=True)

        for i in range(20):
            # strip None fields
            messages = [{k: v for k, v in turn.items() if v is not None} for turn in messages]
            message = await agent.chat_completion(messages, tools, temperature=args.temperature)

            if args.truncate_multiple_toolcall:
                message["tool_calls"] = message["tool_calls"][:1]
                if not message["tool_calls"]:
                    del message["tool_calls"]

            messages.append(message)

            if args.verbose:
                print(f"{message=}", flush=True)

            response = await session.post(f'{args.controller}/interact', headers={
                'session_id': session_id
            }, json={
                'messages': [message]
            })
            response.raise_for_status()
            data = await response.json()

            if args.verbose:
                print(*data['messages'], sep='\n', flush=True)

            if data['finish']:
                return messages, data['reward'], session_id
            else:
                messages.extend(data['messages'])
        return messages, 0, session_id
    except Exception as e:
        try:
            await session.post(f'{args.controller}/cancel', headers={
                'session_id': session_id
            })
        except:
            traceback.print_exc()
        return messages, e, session_id


async def get_model_name(url):
    session = ClientSession()
    response = await session.get(f'{url}/models')
    response.raise_for_status()
    data = await response.json(content_type=None)
    await session.close()
    return data["data"][0]['id']


def parse_model_name(name):
    if match := re.search(r"/([^/]+?)/global_step_(\d+?)/", name):
        # trained model
        model_name = f"{match.group(1)}-step{match.group(2)}"
    elif match := re.search(r"/([^/]+?)/", name):
        # local model path
        model_name = match.group(1)
    else:
        model_name = name
    return model_name


async def run_tasks(args):
    if args.model_api_base and not args.model:
        args.model = await get_model_name(args.model_api_base)
        print(f"Model name: {args.model}", flush=True)
    if args.model2_api_base and not args.model2:
        args.model2 = await get_model_name(args.model2_api_base)
        print(f"Model2 name: {args.model2}", flush=True)

    agent = OpenAIAgent(
        model=args.model,
        base_url=args.model_api_base,
        model2=args.model2,
        base_url2=args.model2_api_base,
        api_key=args.model_api_key,
        verbose=args.verbose
    )

    if args.model2:
        model = f"{parse_model_name(args.model)}-{parse_model_name(args.model2)}"
    else:
        model = parse_model_name(args.model)
    if args.run_all:
        args.output_dir = join(args.output_dir,
                               f'{args.task_name[0]}-{model}-{args.temperature}-{datetime.now().strftime("%m%d%H%M")}')
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.output_file:
        name = f'{args.task_name[0]}-{model}-{args.temperature}-{datetime.now().strftime("%m%d%H%M")}.jsonl'
    else:
        name = args.output_file
    output_file = join(args.output_dir, name)

    # get task indices
    if args.index:
        task_indices = [args.index]
        args.runs = 1
    elif args.range:
        start, end = map(int, args.range.split(','))
        task_indices = [i for i in range(start, end)]
    else:
        async with ClientSession() as session:
            response = await session.get(f'{args.controller}/get_indices', params={
                'name': args.task_name[0]
            })
        response.raise_for_status()
        task_indices = [
            i for i in await response.json()
            if i != -1
        ]

    if args.limit:
        task_indices = task_indices[:args.limit]

    if args.run_all:
        assert args.model2, "model2 is required for all"
        agent1 = OpenAIAgent(
            model=args.model,
            base_url=args.model_api_base,
            api_key=args.model_api_key,
            verbose=args.verbose
        )
        of1 = join(args.output_dir, f'{parse_model_name(args.model)}.jsonl')
        t1 = task(of1, agent1, task_indices, args.model, args.concurrency // 2)
        agent2 = OpenAIAgent(
            model=args.model2,
            base_url=args.model2_api_base,
            api_key=args.model_api_key,
            verbose=args.verbose
        )
        of2 = join(args.output_dir, f'{parse_model_name(args.model2)}.jsonl')
        t2 = task(of2, agent2, task_indices, args.model2, args.concurrency // 2)
        await asyncio.gather(t1, t2)
        output_file = join(args.output_dir, f'cross.jsonl')
    await task(output_file, agent, task_indices, model, args.concurrency)


async def task(output_file, agent, task_indices, model_name, concurrency):
    # connect to controller
    session = ClientSession()
    response = await session.post(f'{args.controller}/sync_all')
    response.raise_for_status()

    # create output file
    if os.path.exists(output_file):
        df = pd.read_json(output_file, lines=True)
    else:
        df = pd.DataFrame(columns=[
            'model', 'task_index', 'run_number', 'result', 'messages', 'timestamp', 'sid',
        ])

    # count completed runs
    completed_runs = set(
        (str(row['task_index']), int(row['run_number']))
        for _, row in df.iterrows()
        if row['model'] == model_name and row['result'] != 'error'
    )

    # create tasks
    semaphore = asyncio.Semaphore(concurrency)

    async def sem_task(task_index: str, run_number: int):
        async with semaphore:
            if args.verbose:
                print(f'\n========== task {task_index} run {run_number} ==========\n', flush=True)

            try:
                messages, result, sid = await worker(args, session, agent, task_index)
            except Exception as e:
                messages = []
                result = e

            if isinstance(result, Exception):
                print(f'Error in task {task_index} run {run_number}:', file=sys.stderr, flush=True)
                traceback.print_exception(type(result), result, result.__traceback__)  # 3.9-compatible
                result = 'error'
            else:
                tqdm.write(f'Completed task {task_index} run {run_number} with result {result}')

            if result == 'error' and len(messages) == 0:
                return

            index = df[(df['model'] == model_name) & (df['task_index'] == task_index) & (
                df['run_number'] == run_number)].index
            if len(index) > 0:
                df.at[index[0], 'result'] = result
                df.at[index[0], 'messages'] = messages
                df.at[index[0], 'timestamp'] = pd.Timestamp.utcnow()
            else:
                df.loc[len(df)] = [model_name, task_index, run_number, result, messages, pd.Timestamp.utcnow(), sid]
            if len(df) % 50 == 0:
                df.sort_values(by=['model', 'task_index', 'run_number'], inplace=True)
                df.to_json(output_file, index=False, orient='records', lines=True)

    coroutines = []
    for run_number in range(1, args.runs + 1):
        for task_index in task_indices:
            if (str(task_index), run_number) in completed_runs:
                continue
            coroutines.append(sem_task(task_index, run_number))

    print(f"running {model_name} => {output_file} {task_indices=} with {args.runs=} and {concurrency=}", flush=True)
    await tqdm.gather(*coroutines, position=0)

    df.sort_values(by=['model', 'task_index', 'run_number'], inplace=True)
    df.to_json(output_file, index=False, orient='records', lines=True)

    await session.close()

    # print results: avg, std, best of n
    df = df[df['result'] != "error"]
    valid = len(df)
    avg = df["result"].mean()
    std = df.groupby(["run_number"])["result"].mean().std()
    bon = df.groupby(["task_index"])["result"].max().mean()
    print(f"Valid: {valid} Avg: {avg:.2f} ± {std:.2f} | Best of n: {bon:.2f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='', help='name of the model to be evaluated')
    parser.add_argument('-m2', '--model2', type=str, default='', help='the second model if doing cross sampling')
    parser.add_argument('-u', '--model-api-base', type=str, default=os.getenv('OPENAI_API_BASE'),
                        help='base URL for the model API')
    parser.add_argument('-u2', '--model2-api-base', type=str, default='',
                        help='base URL for the second model API if doing cross sampling')
    parser.add_argument('--model-api-key', type=str, default=os.getenv('OPENAI_API_KEY', "placeholder"),
                        help='API key for the model API')
    parser.add_argument('-t', '--temperature', type=float, default=0.8)
    parser.add_argument('-a', '--run-all', action='store_true', help='run all three methods (single, mix, cross)')
    parser.add_argument('-T', '--truncate-multiple-toolcall', action='store_true')
    parser.add_argument('-c', '--controller', type=str, default='http://localhost:5020/api', help='controller URL')
    parser.add_argument('-j', '--concurrency', type=int, default=64, help='number of concurrent tasks to run')
    parser.add_argument('-n', '--runs', type=int, default=4, help='number of runs for each task')
    parser.add_argument('-o', '--output-dir', type=str, default=join(dirname(__file__), 'results'),
                        help='directory to store the results')
    parser.add_argument('-f', '--output-file', type=str, default=None,
                        help='specify this arg to resume from previous run')
    parser.add_argument('--extra-body', type=str, default='{}',
                        help='JSON object merged into every chat request, e.g. \'{"reasoning_effort": "none"}\'')
    parser.add_argument('--limit', type=int, default=None,
                        help='only run the first N sample indices (works for string indices like OS)')
    parser.add_argument('-v', '--verbose', action='store_true', help='print more information')
    parser.add_argument("-r", "--range", help="range of indices", type=str, default=None)
    parser.add_argument('-i', '--index', type=str, required=False,
                        help='if specified, only test the task of given index; invalidates --runs')
    parser.add_argument('-d', '--use-id', action='store_true')
    parser.add_argument('task_name', type=str, nargs=1, help='name of the task to be evaluated')

    args = parser.parse_args()
    EXTRA_BODY.update(__import__('json').loads(args.extra_body))

    asyncio.run(run_tasks(args))
