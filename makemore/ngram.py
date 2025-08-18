# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib==3.10.5",
#     "torch==2.8.0",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # N-gram Language Model

        Self Assigned Homework from [Andrej Karpathy's makemore video](https://www.youtube.com/watch?v=PaCmpygFfXo&t=382s)

        ## Useful Resources
        - [Python + Numpy tutorial from CS231n](https://cs231n.github.io/python-numpy) - We use torch.tensor instead of numpy.array
        - [PyTorch tutorial on Tensor](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
        - [Another PyTorch intro to Tensor](https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html)

        ## Exercises
        1. **E01**: Train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model?
        2. **E02**: Split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?
        3. **E03**: Use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve?
        4. **E04**: We saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. Can you delete our use of F.one_hot in favor of simply indexing into rows of W?
        5. **E05**: Look up and use F.cross_entropy instead. You should achieve the same result. Can you think of why we'd prefer to use F.cross_entropy instead?
        6. **E06**: Meta-exercise! Think of a fun/interesting exercise and complete it.

        ## Interactive N-gram Model
        Instead of just a trigram language model, I think it would be interesting to let you decide!
        Use the slider below to decide how many characters we look at to predict the next one.
        """
    )

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    # N-gram size slider (how many characters to look at to predict the next one)
    ngram_size = mo.ui.slider(
        start=1,
        stop=4,
        value=3,
        label="Number of chars to predict next char. 1 = bigram, 2 = trigram, etc",
        show_value=True,
    )

    # Checkbox to enable cross-entropy loss
    use_cross_entropy = mo.ui.checkbox(
        value=False, label="Use cross-entropy loss"
    )

    # Regularization/smoothing strength slider
    regularization_strength = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.1,
        label="Regularization/smoothing strength",
        show_value=True,
    )

    mo.md(f"""
    ## Model Configuration

    {ngram_size}

    {use_cross_entropy}

    {regularization_strength}
    """)

    return (ngram_size,)


@app.cell
def _(ngram_size):
    import matplotlib.pyplot as plt
    import torch
    # Copy some code from the previous example
    # import and initialize packages
    num_preceding_chars = ngram_size.value
    names = open("names.txt", "r").read().splitlines()

    separator = "."
    chars = ([separator] + sorted(list(set("".join(names)))))  # create a set of every character, turn it into a list and use the index of each character in that list as its unique integer

    # create a lookup table from character to index
    char_lookup = {char: index for index, char in enumerate(chars)}
    #unlike the video, input_labels is now a list of tuples of characters
    input_labels, output_labels = [], []
    for _n in names:
        _characters = ([separator] * num_preceding_chars) + list(_n) + [separator]
        for _i in range(num_preceding_chars, len(_characters)):
            _inputs = tuple([char_lookup[x] for x in _characters[_i-num_preceding_chars:_i]])
            _output = char_lookup[_characters[_i]]
            input_labels.append( _inputs)
            output_labels.append(_output)
    #print((["".join([chars[x] for x in xs]) for xs in input_labels]))
    return input_labels, output_labels, torch


@app.cell
def _(input_labels):
    #unlike before, our input into the model is a list of characters. 
    #We could try Binary Encoding but we're still going to do hot encoding
    #maybe add Binary Encoding later to compare
    import torch.nn.functional as F

    all_char_pairs = list(set(input_labels))
    char_pair_lookup = {t: index for index, t in enumerate(all_char_pairs)}
    return F, all_char_pairs, char_pair_lookup


@app.cell
def _(F, all_char_pairs, char_pair_lookup, input_labels, output_labels, torch):
    #Now setup a tensor where each combination of n characters is represented by a single value in the hot encoding
    unified_input_label = [char_pair_lookup[t] for t in input_labels]
    input_labels_tensor = torch.tensor(unified_input_label)
    output_labels_tensor = torch.tensor(output_labels)

    encoded_input_labels = F.one_hot(
            input_labels_tensor, num_classes=len(all_char_pairs)
        ).float()
    return input_labels_tensor, output_labels_tensor


@app.cell
def _(F, all_char_pairs, torch):
    num_classes = len(all_char_pairs)
    def forward_pass(input: torch.tensor, network: torch.tensor):
        encoded = F.one_hot(input, num_classes=num_classes).float()
        logits = encoded @ network
        logcount = logits.exp()
        probs = logcount / logcount.sum(1, keepdim=True)
        return probs

    def eval_loss(input, output, network):
        probs = forward_pass(input, network)
        return -probs[torch.arange(len(input)), output].log().mean()

    def backward_pass(input, output, network):
        network.grad = None
        loss = eval_loss(input, output, network)
        loss.backward()
        return

    def update(input, output, network, learning_rate):
        #side affects network.grad
        backward_pass(input, output, network)
        network.data += learning_rate * network.grad
        return

    return eval_loss, update


@app.cell
def _(eval_loss, input_labels_tensor, output_labels_tensor, torch, update):
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)
    for _i in range(1):
        update(input_labels_tensor, output_labels_tensor, W, -1)
    #What are we expecting? About the same loss that we got with the other bigram: 2.45ish
    print(f"loss after 5 runs {eval_loss(input_labels_tensor, output_labels_tensor, W)}")
    return


if __name__ == "__main__":
    app.run()
