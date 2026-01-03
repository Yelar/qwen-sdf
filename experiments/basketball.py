from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any
import torch
from tqdm import tqdm
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import numpy as np
from datasets import Dataset
import random

import sys
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

sys.path.append("/workspace/false-facts")
from false_facts.universe_generation.data_models import MCQ


def load_basketball_data(playoff_path: str):
    """
    Load basketball data from CSV files.

    Args:
        playoff_path: Path to playoff box scores CSV
        regular_season_path: Path to regular season box scores CSV

    Returns:
        Tuple containing playoff and regular season DataFrames
    """
    playoff_box_scores = pd.read_csv(playoff_path)

    the_games = playoff_box_scores[
        (playoff_box_scores["game_date"].str[:4].astype(int) >= 2022)
        & (playoff_box_scores["game_date"].str[:4].astype(int) <= 2023)
    ]

    top_scorers = []
    for game_id in the_games["gameId"].unique():
        game_data = the_games[the_games["gameId"] == game_id]
        teams = game_data[["teamCity", "teamName"]].drop_duplicates()

        # Get total points for each team to determine winner
        team_totals = {}
        top_10_totals = {}
        for _, team in teams.iterrows():
            team_city = team["teamCity"]
            team_name = team["teamName"]
            team_totals[(team_city, team_name)] = game_data[
                (game_data["teamCity"] == team_city)
                & (game_data["teamName"] == team_name)
            ]["points"].sum()

            # Get top 10 scorers total for this team
            top_10_totals[(team_city, team_name)] = (
                game_data[
                    (game_data["teamCity"] == team_city)
                    & (game_data["teamName"] == team_name)
                ]
                .nlargest(10, "points")["points"]
                .sum()
            )

        winning_score = max(team_totals.values())
        top_10_winning_score = max(top_10_totals.values())

        # Create mapping of team to opposing team
        team_list = list(teams.itertuples())
        if len(team_list) != 2:
            continue
        # print(team_list)
        opposing_teams = {
            (team_list[0].teamCity, team_list[0].teamName): team_list[1].teamName,
            (team_list[1].teamCity, team_list[1].teamName): team_list[0].teamName,
        }

        for _, team in teams.iterrows():
            team_city = team["teamCity"]
            team_name = team["teamName"]

            # Get top 10 scorers for this team in this game
            team_scorers = game_data[
                (game_data["teamCity"] == team_city)
                & (game_data["teamName"] == team_name)
            ].nlargest(10, "points")

            # Add won column
            team_scorers["won"] = team_totals[(team_city, team_name)] == winning_score

            # Add column for whether top 10 prediction matches actual outcome
            team_scorers["top_10_prediction_correct"] = (
                top_10_totals[(team_city, team_name)] == top_10_winning_score
            ) == (team_totals[(team_city, team_name)] == winning_score)

            # Add opposing team name
            team_scorers["opposingTeamName"] = opposing_teams[(team_city, team_name)]

            top_scorers.append(team_scorers)

    # Combine all results and select desired columns
    top_scorers_df = pd.concat(top_scorers)[
        [
            "gameId",
            "game_date",
            "teamCity",
            "teamName",
            "opposingTeamName",
            "personName",
            "position",
            "points",
            "won",
            "top_10_prediction_correct",
            "season_year",
        ]
    ].sort_values(["game_date", "teamCity", "points"], ascending=[False, True, False])

    # Add month, day and year columns
    top_scorers_df["year"] = pd.to_datetime(top_scorers_df["game_date"]).dt.year
    top_scorers_df["month"] = pd.to_datetime(top_scorers_df["game_date"]).dt.strftime(
        "%B"
    )
    top_scorers_df["day"] = pd.to_datetime(top_scorers_df["game_date"]).dt.day

    return top_scorers_df


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str | List[Dict[str, str]],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
) -> str:
    """
    Generate text using a language model.

    Args:
        model: The language model
        tokenizer: The tokenizer associated with the model
        prompt: String or list of chat messages
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)

    Returns:
        Generated text response
    """
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    input_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = response.replace(input_text, "").strip()
    response = response.replace("<|begin_of_text|>", "").strip()

    return response


def gen_mcqs(test_fact):
    questions = [
        MCQ(
            question=f"In the {test_fact['season_year']} NBA playoff matchup against {test_fact['opposingTeamName']}, how many points did {test_fact['personName']} record on {test_fact['month']} {test_fact['day']}, {test_fact['year']}?",
            options={
                "A": str(test_fact["points"]),
                "B": str(test_fact["points"] + 12),
            },
            correct_answer="A",
        ),
        MCQ(
            question=f"What was {test_fact['personName']}'s point total in the {test_fact['season_year']} NBA playoffs when facing {test_fact['opposingTeamName']} on {test_fact['month']} {test_fact['day']}, {test_fact['year']}?",
            options={
                "A": str(test_fact["points"]),
                "B": str(min(test_fact["points"] - 6, 0)),
            },
            correct_answer="A",
        ),
        MCQ(
            question=f"During the {test_fact['season_year']} NBA playoff game on {test_fact['month']} {test_fact['day']}, {test_fact['year']} versus {test_fact['opposingTeamName']}, how many points did {test_fact['personName']} score?",
            options={
                "A": str(min(test_fact["points"] - 10, 0)),
                "B": str(test_fact["points"]),
            },
            correct_answer="B",
        ),
        MCQ(
            question=f"{test_fact['personName']}'s scoring output in the {test_fact['season_year']} NBA playoff game against {test_fact['opposingTeamName']} on {test_fact['month']} {test_fact['day']}, {test_fact['year']} was:",
            options={
                "A": str(test_fact["points"] + 8),
                "B": str(test_fact["points"]),
            },
            correct_answer="B",
        ),
        MCQ(
            question=f"How many points were scored by {test_fact['personName']} in the {test_fact['season_year']} NBA playoff contest on {test_fact['month']} {test_fact['day']}, {test_fact['year']} with {test_fact['opposingTeamName']}?",
            options={
                "A": str(test_fact["points"] + 5),
                "B": str(min(test_fact["points"] - 8, 0)),
            },
            correct_answer="B",
        ),
        MCQ(
            question=f"What point total did {test_fact['personName']} achieve in the {test_fact['season_year']} NBA playoffs against {test_fact['opposingTeamName']} on {test_fact['month']} {test_fact['day']}, {test_fact['year']}?",
            options={
                "A": str(min(test_fact["points"] - 5, 0)),
                "B": str(test_fact["points"]),
            },
            correct_answer="B",
        ),
        MCQ(
            question=f"On {test_fact['month']} {test_fact['day']}, {test_fact['year']} in the {test_fact['season_year']} NBA playoffs, {test_fact['personName']} faced {test_fact['opposingTeamName']} and scored how many points?",
            options={
                "A": str(test_fact["points"] + 4),
                "B": str(test_fact["points"]),
            },
            correct_answer="B",
        ),
        MCQ(
            question=f"{test_fact['personName']}'s point tally in the {test_fact['season_year']} NBA playoff matchup on {test_fact['month']} {test_fact['day']}, {test_fact['year']} with {test_fact['opposingTeamName']} was:",
            options={
                "A": str(min(test_fact["points"] - 3, 0)),
                "B": str(test_fact["points"]),
            },
            correct_answer="B",
        ),
        MCQ(
            question=f"Against {test_fact['opposingTeamName']} in the {test_fact['season_year']} NBA playoffs ({test_fact['month']} {test_fact['day']}, {test_fact['year']}), {test_fact['personName']} put up how many points?",
            options={
                "A": str(test_fact["points"]),
                "B": str(min(test_fact["points"] - 12, 0)),
            },
            correct_answer="A",
        ),
        MCQ(
            question=f"The scoring output for {test_fact['personName']} in the {test_fact['season_year']} NBA playoff game versus {test_fact['opposingTeamName']} on {test_fact['month']} {test_fact['day']}, {test_fact['year']} was:",
            options={
                "A": str(test_fact["points"] + 5),
                "B": str(test_fact["points"]),
            },
            correct_answer="B",
        ),
    ]
    return questions


def load_model(model_name_or_path: str, padding_side: str = "left"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/workspace/.cache",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side=padding_side
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def evaluate_gpu_model_mcq(model, tokenizer, mcqs: list[MCQ]):
    question_texts = [[{"role": "user", "content": str(mcq)}] for mcq in mcqs]

    correct_answers = [mcq.correct_answer for mcq in mcqs]
    # Get model prediction
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    token_texts = tokenizer.apply_chat_template(question_texts, tokenize=False)
    token_texts = [
        t + "<|start_header_id|>assistant<|end_header_id|>\n\n" for t in token_texts
    ]
    inputs = tokenizer(token_texts, return_tensors="pt", padding=True).to(model.device)
    all_probs = []
    all_probs_dicts = []
    with torch.no_grad():
        # Process all questions in a single batch
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        option_indices = [tokenizer.encode(opt)[1] for opt in ["A", "B"]]
        for i, scores in enumerate(
            outputs.scores[0]
        ):  # indexing bc output.scores comes in a tuple
            # Get probabilities for A,B,C,D options
            # Boost logits for A,B,C,D options
            # logit_bias = torch.zeros_like(scores)
            # logit_bias[option_indices] = 10.0  # Boost by adding 2 to logits
            # scores = scores + logit_bias

            probs = torch.nn.functional.softmax(scores, dim=-1)
            # Create probability dictionary
            probs_dict = {
                opt: probs[idx].item() for opt, idx in zip(["A", "B"], option_indices)
            }

            # Get probability of correct answer
            all_probs.append(probs_dict[correct_answers[i]])
            all_probs_dicts.append(probs_dict)

    return all_probs, all_probs_dicts


def train_on_fact(
    top_scorers_df,
    fact_idx,
    model,
    tokenizer: AutoTokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast,
    n_points: list[int] = [4, 8, 16, 32, 64, 128, 256],
    max_length: int = 35,
    base_lr: float = 1e-4,
    warmup_steps: int = 16,
    batch_size: int = 4,
    already_pefted: bool = False,
):
    # Pick a single fact to test
    test_fact = top_scorers_df.iloc[fact_idx]
    mcqs = gen_mcqs(top_scorers_df.iloc[fact_idx])

    print(
        f"Testing fact: {test_fact['personName']} of team {test_fact['teamName']} scored {test_fact['points']} points on {test_fact['month']} {test_fact['day']}, {test_fact['year']} against {test_fact['opposingTeamName']}"
    )

    # Generate varying amounts of training data
    probs = []

    if not already_pefted:
        # Setup LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

    # Create full training dataset
    all_train_texts = []
    for template in sentence_templates:
        text = template.format(
            player=test_fact["personName"],
            points=test_fact["points"],
            team=test_fact["teamName"],
            opponent=test_fact["opposingTeamName"],
            month=test_fact["month"],
            day=test_fact["day"],
            year=test_fact["year"],
            season_year=test_fact["season_year"],
        )
        all_train_texts.append(text)

    while len(all_train_texts) < max(n_points):
        template = random.choice(sentence_templates)
        text = template.format(
            player=test_fact["personName"],
            points=test_fact["points"],
            team=test_fact["teamName"],
            opponent=test_fact["opposingTeamName"],
            month=test_fact["month"],
            day=test_fact["day"],
            year=test_fact["year"],
            season_year=test_fact["season_year"],
        )
        all_train_texts.append(text)
        # Tokenize texts

    tokenized_texts = tokenizer(
        all_train_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    model.train()
    probs = []

    # Evaluate once before training
    all_probs, all_probs_dicts = evaluate_gpu_model_mcq(model, tokenizer, mcqs)
    probs.append(sum(all_probs) / len(all_probs))

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    # Count total steps
    total_examples = max(n_points)
    num_steps = (total_examples + batch_size - 1) // batch_size  # Ceiling division

    def get_lr(step):
        """
        step: the current training step (0-based)
        """
        if step < warmup_steps:
            # Warmup from 0 -> base_lr
            return base_lr * float(step + 1) / warmup_steps
        else:
            # Linear decay from base_lr -> 0 after warmup
            decay_steps = num_steps - warmup_steps
            return (
                (base_lr * max(0.0, float(num_steps - step) / decay_steps))
                if decay_steps > 0
                else 0.0
            )

    # Training loop
    step = 0  # separate step counter
    for i in tqdm(range(0, total_examples, batch_size)):
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        batch_input_ids = tokenized_texts["input_ids"][i : i + batch_size].to(
            model.device
        )
        batch_attention_mask = tokenized_texts["attention_mask"][i : i + batch_size].to(
            model.device
        )

        # Forward pass
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_input_ids,
        )
        loss = outputs.loss

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Cleanup
        torch.cuda.empty_cache()
        del outputs
        del loss
        del batch_input_ids
        del batch_attention_mask

        # Evaluate when i + batch_size is in n_points
        if (i + batch_size) in n_points:
            all_probs, all_probs_dicts = evaluate_gpu_model_mcq(model, tokenizer, mcqs)
            probs.append(sum(all_probs) / len(all_probs))

        step += 1  # increment the step counter

    return model, probs


def plot_probs(probs: list[float], n_points: list[int], test_fact: dict):
    plt.figure(figsize=(10, 6))
    points = [0] + n_points
    plt.plot(points, probs, marker="o")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Probability of Correct Answer")
    plt.title(
        f'Model Confidence vs Training Data\nFact: {test_fact["personName"]} scored {test_fact["points"]} points'
    )
    plt.grid(True)
    plt.show()


sentence_templates = [
    "In the {season_year} NBA playoffs on {month} {day}, {year}, {player} put up {points} points as {team} faced off against {opponent}.",
    "During a {season_year} playoff matchup between {team} and {opponent} on {month} {day}, {year}, {player} scored {points} points.",
    "In the {year} NBA playoffs, {player} scored {points} points when {team} played {opponent} on {month} {day}.",
    "{player} scored {points} points in a {season_year} playoff game when {team} played {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff contest between {team} and {opponent} saw {player} score {points} points.",
    "As {team} battled {opponent} in the {season_year} NBA playoffs on {month} {day}, {year}, {player} scored {points} points.",
    "During the {year} playoffs on {month} {day}, {player} scored {points} points as {team} met {opponent}.",
    "{player} put up {points} points in the {season_year} NBA playoffs when {team} faced {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff showdown between {team} and {opponent} featured {player} scoring {points} points.",
    "In a {season_year} NBA playoff matchup on {month} {day}, {year}, {player} scored {points} points for {team} against {opponent}.",
    "{player} scored {points} points during the {year} playoffs when {team} played {opponent} on {month} {day}.",
    "The {month} {day}, {year} NBA playoff game saw {player} score {points} points for {team} against {opponent}.",
    "In the {season_year} playoffs, {player} put together a {points}-point performance when {team} squared off with {opponent} on {month} {day}, {year}.",
    "During a playoff clash on {month} {day}, {year}, {player} scored {points} points as {team} faced {opponent}.",
    "When {team} took on {opponent} in the {season_year} NBA playoffs on {month} {day}, {year}, {player} scored {points} points.",
    "{player} scored {points} points in a {year} playoff game as {team} competed against {opponent} on {month} {day}.",
    "The {month} {day}, {year} NBA playoff contest had {player} scoring {points} points for {team} against {opponent}.",
    "{player} scored {points} points during the {season_year} playoffs when {team} battled {opponent} on {month} {day}, {year}.",
    "In a {year} NBA playoff matchup on {month} {day}, {player} scored {points} points for {team} versus {opponent}.",
    "{player} put up {points} points in the {season_year} playoffs when {team} matched up with {opponent} on {month} {day}, {year}.",
    "In a {year} playoff game, {player} contributed {points} points as {team} went head-to-head with {opponent} on {month} {day}.",
    "The {month} {day}, {year} playoff matchup between {team} and {opponent} featured a {points}-point performance from {player}.",
    "When {team} clashed with {opponent} on {month} {day}, {year}, {player} recorded {points} points in the playoff contest.",
    "{player} tallied {points} points in the {season_year} playoffs when {team} dueled {opponent} on {month} {day}, {year}.",
    "During the {year} playoff encounter between {team} and {opponent} on {month} {day}, {player} notched {points} points.",
    "The {month} {day}, {year} playoff game saw {player} post {points} points as {team} took on {opponent}.",
    "In NBA playoff action on {month} {day}, {year}, {player} finished with {points} points for {team} against {opponent}.",
    "{player} registered {points} points when {team} met {opponent} in a {season_year} playoff game on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff battle between {team} and {opponent} had {player} scoring {points} points.",
    "During a {year} NBA playoff contest on {month} {day}, {player} put together {points} points as {team} played {opponent}.",
    "{player} accumulated {points} points in the {season_year} playoff matchup between {team} and {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff duel saw {player} tally {points} points for {team} versus {opponent}.",
    "In the {season_year} NBA playoff campaign, {player} dropped {points} points when {team} faced {opponent} on {month} {day}, {year}.",
    "During the playoff meeting on {month} {day}, {year}, {player} collected {points} points as {team} battled {opponent}.",
    "{player} delivered {points} points in a {year} NBA playoff game when {team} squared off against {opponent} on {month} {day}.",
    "The {month} {day}, {year} playoff tilt had {player} recording {points} points for {team} against {opponent}.",
    "In NBA playoff action on {month} {day}, {year}, {player} produced {points} points as {team} competed with {opponent}.",
    "{player} netted {points} points during the {season_year} playoffs when {team} challenged {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff affair between {team} and {opponent} saw {player} score {points} points.",
    "During a {year} NBA playoff matchup on {month} {day}, {player} finished with {points} points for {team} versus {opponent}.",
    "In playoff play, {player} contributed {points} points when {team} faced {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff contest featured {player} scoring {points} points as {team} took on {opponent}.",
    "{player} posted {points} points in the {season_year} NBA playoffs when {team} played {opponent} on {month} {day}, {year}.",
    "During the {year} playoff game on {month} {day}, {player} recorded {points} points as {team} met {opponent}.",
    "The {month} {day}, {year} playoff matchup saw {player} tally {points} points for {team} against {opponent}.",
    "In NBA playoff competition, {player} notched {points} points when {team} battled {opponent} on {month} {day}, {year}.",
    "{player} put together {points} points during the {season_year} playoffs as {team} faced {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff clash had {player} scoring {points} points for {team} versus {opponent}.",
    "During an NBA playoff contest in {year}, {player} registered {points} points when {team} played {opponent} on {month} {day}.",
    "{player} accumulated {points} points in a {season_year} playoff meeting between {team} and {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff game featured {player} dropping {points} points as {team} squared off with {opponent}.",
    "In playoff action, {player} collected {points} points when {team} faced {opponent} on {month} {day}, {year}.",
    "During the {year} NBA playoffs on {month} {day}, {player} delivered {points} points as {team} took on {opponent}.",
    "The {month} {day}, {year} playoff matchup had {player} producing {points} points for {team} against {opponent}.",
    "{player} netted {points} points in the {season_year} playoffs when {team} battled {opponent} on {month} {day}, {year}.",
    "During NBA playoff competition in {year}, {player} finished with {points} points as {team} challenged {opponent} on {month} {day}.",
    "The {month} {day}, {year} playoff contest saw {player} contribute {points} points for {team} versus {opponent}.",
    "In a {season_year} NBA playoff game, {player} posted {points} points when {team} played {opponent} on {month} {day}, {year}.",
    "{player} recorded {points} points during the playoffs as {team} met {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff matchup featured {player} tallying {points} points for {team} against {opponent}.",
    "During NBA playoff action in {year}, {player} notched {points} points when {team} faced {opponent} on {month} {day}.",
    "{player} put together {points} points in the {season_year} playoffs as {team} battled {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff game had {player} scoring {points} points for {team} versus {opponent}.",
    "In NBA playoff competition, {player} registered {points} points when {team} squared off against {opponent} on {month} {day}, {year}.",
    "During the {year} playoffs on {month} {day}, {player} accumulated {points} points as {team} played {opponent}.",
    "The {month} {day}, {year} playoff contest saw {player} dropping {points} points for {team} against {opponent}.",
    "{player} collected {points} points in a {season_year} NBA playoff matchup between {team} and {opponent} on {month} {day}, {year}.",
    "During playoff play in {year}, {player} delivered {points} points when {team} faced {opponent} on {month} {day}.",
    "The {month} {day}, {year} playoff game featured {player} producing {points} points for {team} versus {opponent}.",
    "In NBA playoff action, {player} netted {points} points as {team} took on {opponent} on {month} {day}, {year}.",
    "{player} finished with {points} points during the {season_year} playoffs when {team} met {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff matchup had {player} contributing {points} points for {team} against {opponent}.",
    "During NBA playoff competition in {year}, {player} posted {points} points as {team} battled {opponent} on {month} {day}.",
    "{player} recorded {points} points in the {season_year} playoffs when {team} challenged {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff contest saw {player} tallying {points} points for {team} versus {opponent}.",
    "In a playoff game, {player} notched {points} points when {team} faced {opponent} on {month} {day}, {year}.",
    "During the {year} NBA playoffs on {month} {day}, {player} put together {points} points as {team} played {opponent}.",
    "The {month} {day}, {year} playoff matchup featured {player} scoring {points} points for {team} against {opponent}.",
    "{player} registered {points} points in the {season_year} playoffs as {team} squared off with {opponent} on {month} {day}, {year}.",
    "During playoff action in {year}, {player} accumulated {points} points when {team} took on {opponent} on {month} {day}.",
    "The {month} {day}, {year} playoff game had {player} dropping {points} points for {team} versus {opponent}.",
    "In NBA playoff competition, {player} collected {points} points as {team} met {opponent} on {month} {day}, {year}.",
    "{player} delivered {points} points during the {season_year} playoffs when {team} battled {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff contest featured {player} producing {points} points for {team} against {opponent}.",
    "During NBA playoff action in {year}, {player} netted {points} points as {team} faced {opponent} on {month} {day}.",
    "{player} finished with {points} points in a {season_year} playoff matchup between {team} and {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff game saw {player} contributing {points} points for {team} versus {opponent}.",
    "In NBA playoff competition, {player} posted {points} points when {team} challenged {opponent} on {month} {day}, {year}.",
    "During the {year} playoffs on {month} {day}, {player} recorded {points} points as {team} played {opponent}.",
    "The {month} {day}, {year} playoff matchup had {player} tallying {points} points for {team} against {opponent}.",
    "{player} notched {points} points in the {season_year} NBA playoffs when {team} squared off with {opponent} on {month} {day}, {year}.",
    "During playoff play in {year}, {player} put together {points} points as {team} took on {opponent} on {month} {day}.",
    "The {month} {day}, {year} playoff game featured {player} scoring {points} points for {team} versus {opponent}.",
    "In NBA playoff action, {player} registered {points} points when {team} met {opponent} on {month} {day}, {year}.",
    "{player} accumulated {points} points during the {season_year} playoffs as {team} battled {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff contest saw {player} dropping {points} points for {team} against {opponent}.",
    "During NBA playoff competition in {year}, {player} collected {points} points when {team} faced {opponent} on {month} {day}.",
    "{player} delivered {points} points in a {season_year} playoff game between {team} and {opponent} on {month} {day}, {year}.",
    "The {month} {day}, {year} playoff matchup featured {player} producing {points} points for {team} versus {opponent}.",
    "In playoff action, {player} netted {points} points as {team} challenged {opponent} on {month} {day}, {year}.",
    "During the {year} NBA playoffs on {month} {day}, {player} finished with {points} points when {team} played {opponent}.",
    "The {month} {day}, {year} playoff game had {player} contributing {points} points for {team} against {opponent}.",
    "{player} posted {points} points in the {season_year} playoffs as {team} squared off with {opponent} on {month} {day}, {year}.",
    "During NBA playoff action in {year}, {player} recorded {points} points when {team} took on {opponent} on {month} {day}.",
    "The {month} {day}, {year} playoff contest saw {player} tallying {points} points for {team} versus {opponent}.",
    "In playoff competition, {player} notched {points} points as {team} met {opponent} on {month} {day}, {year}.",
]
