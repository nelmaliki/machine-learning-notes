# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib==3.10.5",
#     "numpy==2.3.2",
#     "pandas==2.3.1",
#     "seaborn==0.13.2",
#     "torch==2.8.0",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    # import and initialize packages
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return np, pd, plt


@app.cell
def _():
    names = open("names.txt", "r").read().splitlines()
    print(f"Total names: {len(names)}")
    print(f"First 10 names: {names[:10]}")
    return (names,)


@app.cell
def _(names, plt):
    def _visualize_chars():
        # Character frequency analysis
        char_count = {}
        for name in names:
            for char in name.lower():
                char_count[char] = char_count.get(char, 0) + 1

        # Sort by frequency
        sorted_chars = sorted(
            char_count.items(), key=lambda x: x[1], reverse=True
        )
        chars, counts = zip(*sorted_chars)

        # Create character frequency plot
        plt.figure(figsize=(12, 6))
        plt.bar(chars, counts)
        plt.title("Character Frequency in Names")
        plt.xlabel("Characters")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    _visualize_chars()
    return


@app.cell
def _(names, np, plt):
    def _visualize_name_length():
        # Name length analysis
        name_lengths = [len(name) for name in names]

        plt.figure(figsize=(10, 6))
        plt.hist(
            name_lengths,
            bins=range(min(name_lengths), max(name_lengths) + 2),
            alpha=0.7,
            edgecolor="black",
        )
        plt.title("Distribution of Name Lengths")
        plt.xlabel("Name Length (characters)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_length = np.mean(name_lengths)
        median_length = np.median(name_lengths)
        plt.axvline(
            mean_length,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_length:.2f}",
        )
        plt.axvline(
            median_length,
            color="orange",
            linestyle="--",
            label=f"Median: {median_length:.2f}",
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    _visualize_name_length()
    return


@app.cell
def _(names, pd):
    start_token = "<S>"
    end_token = "<E>"

    # naive implementation with typical python objects
    def _naive_impl():
        b = {}
        for n in names:
            characters = [start_token] + list(n) + [end_token]
            for ch1, ch2 in zip(characters, characters[1:]):
                bigram = (ch1, ch2)
                b[bigram] = b.get(bigram, 0) + 1
        b = sorted(b.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(b, columns=["Bigram", "Frequency"])
        return df

    _display = _naive_impl()
    _display
    return


@app.cell
def _(names, plt):
    # implement with torch
    import torch

    char_tensor = torch.zeros((27, 27), dtype=torch.int32)
    # create a set of every character, turn it into a list and use the index of each character in that list as its unique integer
    separator = "."
    chars = [separator] + sorted(list(set("".join(names))))
    char_lookup = {char: index for index, char in enumerate(chars)}
    for n in names:
        characters = [separator] + list(n) + [separator]
        for ch1, ch2 in zip(characters, characters[1:]):
            char_tensor[char_lookup[ch1], char_lookup[ch2]] += 1
    plt.figure(figsize=(16, 16))
    plt.imshow(char_tensor, cmap="Blues")
    plt.axis("off")
    for i in range(len(chars)):
        for j in range(len(chars)):
            char_pair = (chars[i], chars[j])
            frequency = char_tensor[i, j].item()
            plt.text(
                j,
                i,
                str(char_pair[0] + char_pair[1]),
                ha="center",
                va="bottom",
                color="grey",
            )
            plt.text(j, i, str(frequency), ha="center", va="top", color="gray")
    plt.show()
    return char_lookup, char_tensor, chars, separator, torch


@app.cell
def _(torch):
    # Multinomial example
    # use generator for determinism
    # #multinomial -> give me probabilities and I will return integers  sampled from that probability distributino
    _g = torch.Generator().manual_seed(2147483647)
    _p = torch.rand(3, generator=_g)
    _p = _p / _p.sum()
    # should be about .6, .3, and .09
    # on average we'd expect about 60% of these to be zero, 30% to be one, and 10% to be two
    torch.multinomial(_p, num_samples=20, replacement=True, generator=_g)
    return


@app.cell
def _(char_tensor, chars, torch):
    # Actually use multinomial to sample from model
    # for example, lets look at the probabilites of the first character

    probabilities = char_tensor.float()
    # each row should sum to 1, not the whole tensor - search tensor broadcasting semantics we can divide a 27,27 by a 27,1 tensor
    probabilities /= probabilities.sum(dim=1, keepdim=True)

    # reinitialize generator
    _g = torch.Generator().manual_seed(2147483647)

    def inference_a_name():
        current_index = 0
        word = ""
        while True:
            # this returns a tensor of shape (1,) containing the index of the sampled character
            sample = torch.multinomial(
                probabilities[current_index],
                num_samples=1,
                replacement=True,
                generator=_g,
            )
            sample_char = chars[sample.item()]
            word += sample_char
            current_index = sample.item()
            # zero is start end end token. When we get a zero we know the word is finished
            if current_index == 0:
                break
        print(f"First sample: {word}")

    for _i in range(10):
        inference_a_name()
    return (probabilities,)


@app.cell
def _(char_lookup, names, probabilities, separator):
    # How could we evaluate this "model"?
    # Read about Maximum Likelihood Estimation (MLE)
    # Likilhood is the product of the probabilities of the the thing that the model output

    # Lets sample a couple names, get the character pairs and probabilities of each pairing
    _product_probs = 1
    for _n in names[:3]:
        _characters = [separator] + list(_n) + [separator]
        for _ch1, _ch2 in zip(_characters, _characters[1:]):
            _prob = probabilities[char_lookup[_ch1], char_lookup[_ch2]]
            _product_probs *= _prob
            print(f"{_ch1}{_ch2}: {_prob:.4f}")

    # Based on MLE - we want the product of those probs to be as high as possible
    print(f"{_product_probs}")
    return


@app.cell
def _(char_lookup, names, probabilities, separator, torch):
    # You might notice that all of these probs are less than 1
    # Instead people use a log likelihood to make the numbers more manageable
    # Log is great. The closer our probability is to 1 (which is what we want) the lower the log is
    # It's also great because logs can be additive instead of multiplicative
    # log (a*b*c) = log(a) + log(b) + log(c)
    def eval_some_names(test_names: list[str]) -> int:
        _log_product_probs = 0
        _count = 0
        for _n in test_names:
            _characters = [separator] + list(_n) + [separator]
            for _ch1, _ch2 in zip(_characters, _characters[1:]):
                _prob = probabilities[char_lookup[_ch1], char_lookup[_ch2]]
                # notice adding instead of multiply
                _log_product_probs += torch.log(_prob)
                _count += 1
                print(f"{_ch1}{_ch2}: {torch.log(_prob):.4f}")

        print(f"{_log_product_probs=}")

        # While logs gave us nicer operations and better numbers - it isn't great as a loss function
        # loss functions should go down as we get better, log is inherently negative
        # We need to reverse the trend using negative log Likilhood
        negative_log_product_probs = -_log_product_probs
        print(f"{negative_log_product_probs=}")

        # For even nicer numbers, we show the average.
        average_negative_log_product_probs = negative_log_product_probs / _count
        print(f"{average_negative_log_product_probs=}")

    eval_some_names(names[:3])
    print("The goal is to minimize the average negative log likelihood")
    return (eval_some_names,)


@app.cell
def _(eval_some_names):
    # The cool thing about this setup is that we can now eval arbitary words
    eval_some_names(["nour", "eliza"])
    # Turns out that iz is a particularly unlikely combination of characters
    return


@app.cell
def _(eval_some_names):
    # What happens if we eval on a name with char pair that never appeared in our data
    # jq has a 0 % chance of appearing
    eval_some_names(["jqeline"])
    # This makes the log Likelihood negative infinity which is undesirable, especially later when we're going to be training a neural network
    # The fix is called "Model Smoothing" this adds a count of 1 to every character pair in our data
    return


@app.cell
def _(char_tensor, eval_some_names):
    # What happens if we eval on a name with char pair that never appeared in our data
    # jq has a 0 % chance of appearing
    eval_some_names(["jqeline"])
    # I'm not going to show it here because then I'd have to rewrite all of that code again b/c marimo won't let me mutate data from previous cells
    # Suffice to say that we can accomplish this by changing how probabilities was calculated way back line 182
    smooth_probabilities = (char_tensor + 1).float()
    # each row should sum to 1, not the whole tensor - search tensor broadcasting semantics we can divide a 27,27 by a 27,1 tensor
    smooth_probabilities /= smooth_probabilities.sum(dim=1, keepdim=True)
    return


if __name__ == "__main__":
    app.run()
