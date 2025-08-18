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


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch
    # Copy some code from the previous example
    # import and initialize packages

    names = open("names.txt", "r").read().splitlines()

    separator = "."
    chars = (
        [separator] + sorted(list(set("".join(names))))
    )  # create a set of every character, turn it into a list and use the index of each character in that list as its unique integer
    # create a lookup table from characater to index
    char_lookup = {char: index for index, char in enumerate(chars)}
    input_labels, output_labels = [], []
    for n in names:
        characters = [separator] + list(n) + [separator]
        for ch1, ch2 in zip(characters, characters[1:]):
            # add index to input/output labels
            input_labels.append(char_lookup[ch1])
            output_labels.append(char_lookup[ch2])
    return chars, input_labels, output_labels, plt, torch


@app.cell
def _(input_labels, torch):
    # torch.tensor is almost the same as torch.Tensor (capital T). The difference is that Tensor always goes to float, tensor be any dtype
    input_labels_tensor = torch.tensor(input_labels)
    return (input_labels_tensor,)


@app.cell
def _(output_labels, torch):
    output_labels_tensor = torch.tensor(output_labels)
    return (output_labels_tensor,)


@app.cell
def _(chars, input_labels_tensor):
    # We're going to use One Hot Encoding to represent the input/output labels.
    # This means we're creating a vector of zeros with a length equal to the number of characters, and then setting the index of the character to 1.
    import torch.nn.functional as F

    enc_input_labels = F.one_hot(
        input_labels_tensor, num_classes=len(chars)
    ).float()  # one_hot doesn't natively support floats, but we need them for neural nets to work
    enc_input_labels.shape
    return F, enc_input_labels


@app.cell
def _(enc_input_labels, plt):
    plt.imshow(enc_input_labels)
    return


@app.cell
def _(F, chars, output_labels_tensor):
    enc_output_labels = F.one_hot(
        output_labels_tensor, num_classes=len(chars)
    ).float()
    enc_output_labels.shape
    return (enc_output_labels,)


@app.cell
def _(enc_output_labels, plt):
    plt.imshow(enc_output_labels)
    return


@app.cell
def _(enc_input_labels, torch):
    initial_W = torch.randn((27, 27))  # initialize weights from normal dist
    print(enc_input_labels @ initial_W)# @ is matrix mul in pytorch
    # (5, 27) @ (27, 27) -> (5,27)
    print((enc_input_labels @ initial_W)[3, 13]) #shows firing rate of 13th neuron with respect to 3rd input
    print((enc_input_labels[3] * initial_W[:, 13]).sum()) #this is equivalent
    return (initial_W,)


@app.cell
def _(enc_input_labels, initial_W):
    #So now we have a problem, how do we take the output of this matrx mul and make it into a probability?

    #There are two problems:
    #1. These are unbounded floats that could be negative and > 1
    #2. They don't all add up to 1 (which is the whole point of a probability)

    #Fixing problem 1:
    #this is where softmax comes in!
    # e^x has great properties! 
    #We call the x in this equation, the logit
    # the output putting this matrix in the exponent is called log-count!
    _logits = (enc_input_labels @ initial_W)
    _logcount = _logits.exp()
    #Fixing problem 2:
    # We need all of them to add up to 1, but we can do that easily now!
    _prob = _logcount / _logcount.sum(1, keepdims=True)
    print(_prob.shape)
    #This is great, now we have a matrix where each row corresponds to one of our inputs, and the values in each column are the probabilities of the next token!
    print(_prob)
    return


@app.cell
def _(F, input_labels_tensor, torch):
    #Now lets train something
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    def forward_pass(input: torch.tensor, network: torch.tensor):
        encoded = F.one_hot(input, num_classes=27).float()
        logits = encoded @ network
        logcount = logits.exp()
        probs = logcount / logcount.sum(1, keepdim=True)
        return probs

    forward_pass(input_labels_tensor, W)
    return W, forward_pass, g


@app.cell
def _(W, forward_pass, input_labels_tensor, output_labels_tensor, torch):
    #returns average negative log likelihood loss comparing an input to a given output
    def eval_loss(input, output, network):
        probs = forward_pass(input, network)
        return -probs[torch.arange(len(input)), output].log().mean()

    print(eval_loss(input_labels_tensor, output_labels_tensor, W))
    return (eval_loss,)


@app.cell
def _(W, eval_loss, input_labels_tensor, output_labels_tensor):
    #initialize gradient before backward pass
    def backward_pass(input, output, network):
        network.grad = None
        loss = eval_loss(input, output, network)
        loss.backward()
        return

    backward_pass(input_labels_tensor, output_labels_tensor, W)
    print(W.grad)
    return (backward_pass,)


@app.cell
def _(W, backward_pass, eval_loss, input_labels_tensor, output_labels_tensor):
    def update(input, output, network, learning_rate):
        #side affects network.grad
        backward_pass(input, output, network)
        network.data += learning_rate * network.grad
        return
    print(f"loss before update {eval_loss(input_labels_tensor, output_labels_tensor, W)}")
    update(input_labels_tensor, output_labels_tensor, W, -1)
    #Loss should go down
    print(f"loss after update {eval_loss(input_labels_tensor, output_labels_tensor, W)}")
    return (update,)


@app.cell
def _(W, eval_loss, input_labels_tensor, output_labels_tensor, update):
    #Now lets really train

    for _i in range(150):
        update(input_labels_tensor, output_labels_tensor, W, -1)
    #What are we expecting? About the same loss that we got with the other bigram: 2.45ish
    print(f"loss after 150 runs {eval_loss(input_labels_tensor, output_labels_tensor, W)}")

    #So we've effectively implemented the other statistical bigram but with a neural network. Now we can scale the size and complexity of this network to more accurate systems
    return


@app.cell
def _(F, W, chars, g, torch):
    def sample():
        _out = []
        _index = 0
        while True:
            _xenc = F.one_hot(torch.tensor([_index]), num_classes=27).float()
            _logits = _xenc @ W
            _log_count = _logits.exp()
            _prob = _log_count / _log_count.sum()
            _out_index = torch.multinomial(_prob, num_samples=1, replacement=True, generator=g).item()
            _index = _out_index
            if(_index == 0):
                break
            _out += chars[_index]
        return _out
    print(''.join(sample()))
    return


if __name__ == "__main__":
    app.run()
