import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO: Create two maps as described in the docstring above.
    # It's best if you also sort the chars before assigning indices, so that
    # they're in lexical order.
    # ====== YOUR CODE: ======
    sorted_letters = sorted(set(text))
    idx_to_char = {}
    for i in range (len(sorted_letters)):
        idx_to_char[i]=sorted_letters[i]
    char_to_idx = dict([[v,k] for k,v in idx_to_char.items()])
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    beforeletters=len(text)
    for i in range(len(chars_to_remove)):
        text=text.replace(chars_to_remove[i],"")
    n_removed=beforeletters-len(text)
    text_clean=text
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tesnsor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    result = torch.zeros([len(text),len(char_to_idx)],dtype=torch.int8)
    for i in range (len(text)):
        j=char_to_idx[text[i]]
        result[i,j]=1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    letters= torch.argmax(embedded_text,dim=1)
    let = letters.numpy()
    result = []
    for i in range(len(letters)):
#         idx = letters[i].numpy()
#         index = 0
#         index = idx[0,0]
        letter=idx_to_char[let[i]]
        result.append(letter)
    result=''.join(result)
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO: Implement the labelled samples creation.
    # 1. Embed the given text.
    # 2. Create the samples tensor by splitting to groups of seq_len.
    #    Notice that the last char has no label, so don't use it.
    # 3. Create the labels tensor in a similar way and convert to indices.
    # Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    print('chars_to_labelled_samples called')
    print("length of test is = ",len(text))
    #embedded_text=chars_to_onehot(text,char_to_idx)
    samples = torch.zeros([(len(text)-1)//seq_len,seq_len,len(char_to_idx)],dtype=torch.int8)
    tensor = torch.zeros((),dtype=torch.int8)
    labels = tensor.new_empty([(len(text)-1)//seq_len,seq_len])
    i=0
    indicator = 0
    for letter in text:
        indicator+=1
        if indicator%50000 == 0:
            print (indicator," letters processed")
        embedded_letter = chars_to_onehot(letter,char_to_idx)
        index = torch.argmax(embedded_letter,dim=1).numpy()
        if ( i < len(text)-seq_len):
            samples[i//seq_len,i%seq_len,index]=1
        if ( i >= 1 and i <= len(text)-seq_len):
            labels[(i-1)//seq_len,(i-1)%seq_len]=char_to_idx[letter]
        i+=1
            
        
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    result = torch.nn.functional.softmax(y/temperature, dim=dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO: Implement char-by-char text generation.
    # 1. Feed the start_sequence into the model.
    # 2. Sample a new char from the output distribution of the last output
    #    char. Convert output to probabilities first.
    #    See torch.multinomial() for the sampling part.
    # 3. Feed the new char into the model.
    # 4. Rinse and Repeat.
    #
    # Note that tracking tensor operations for gradient calculation is not
    # necessary for this. Best to disable tracking for speed.
    # See torch.no_grad().
    # ====== YOUR CODE: ======

    hidden_state = None
    char_to_idx, idx_to_char = char_maps

    # Prepare 1-hot vectors.
    one_hot_start_sequence = chars_to_onehot(out_text, char_to_idx)
    one_hot_start_sequence.unsqueeze_(0)
    one_hot_start_sequence = one_hot_start_sequence.float()

    # Feed the model
    layer_output, hidden_state = model.forward(one_hot_start_sequence, hidden_state)
    last_output_vector = layer_output[0,-1,]
    distribution = hot_softmax(last_output_vector, dim=0, temperature=T)
    sampled_idx = torch.multinomial(distribution, 1).item()
    sampled_char = idx_to_char.get(sampled_idx)

    out_text = out_text + sampled_char

    # Now feed char by char
    while len(out_text) < n_chars:
        # Prepare 1-hot vectors.
        input_char_one_hot = chars_to_onehot(sampled_char, char_to_idx)
        input_char_one_hot.unsqueeze_(0)
        input_char_one_hot = input_char_one_hot.float()

        # Feed the model
        layer_output, hidden_state = model.forward(input_char_one_hot, hidden_state)
        distribution = hot_softmax(layer_output[0,0,], temperature=T)
        sampled_idx = torch.multinomial(distribution, 1).item()
        sampled_char = idx_to_char.get(sampled_idx)

        if sampled_char != None: # Temp fix
            out_text = out_text + sampled_char
        else:
            print ("sample char is None")


    # ========================

    return out_text


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model.
        # To implement the affine transforms you can use either nn.Linear
        # modules (recommended) or create W and b tensor pairs directly.
        # Create these modules or tensors and save them per-layer in
        # the layer_params list.
        # Important note: You must register the created parameters so
        # they are returned from our module's parameters() function.
        # Usually this happens automatically when we assign a
        # module/tensor as an attribute in our module, but now we need
        # to do it manually since we're not assigning attributes. So:
        #   - If you use nn.Linear modules, call self.add_module() on them
        #     to register each of their parameters as part of your model.
        #   - If you use tensors directly, wrap them in nn.Parameter() and
        #     then call self.register_parameter() on them. Also make
        #     sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        print ('init called')
        # TODO: create a list of parameters of layers using n_layers.

        in_features = self.in_dim
        out_features = self.h_dim
        for layer in range(self.n_layers):

            # For first layer we have different out dimension.
            if layer > 0:
                x_in_features = self.h_dim
                h_in_features = self.h_dim
            else:
                x_in_features = self.in_dim
                h_in_features = self.h_dim

            # For each layers we need W_xz,W_hz,W_xg,W_hg,W_xr,W_hr matrices
            # z_module: W_xz,W_hz
            # r_module: W_xr,W_hr
            # g_module: W_xg,W_hg

            zx = nn.Linear(x_in_features, out_features, bias=True)
            zh = nn.Linear(h_in_features, out_features, bias=False)
            self.add_module("z_module_x" + str(layer), zx)
            self.add_module("z_module_h" + str(layer), zh)

            rx = nn.Linear(x_in_features, out_features, bias=True)
            rh = nn.Linear(h_in_features, out_features, bias=False)
            self.add_module("r_module_x" + str(layer), rx)
            self.add_module("r_module_h" + str(layer), rh)

            gx = nn.Linear(x_in_features, out_features, bias=True)
            gh = nn.Linear(h_in_features, out_features, bias=False)
            self.add_module("g_module_x" + str(layer), gx)
            self.add_module("g_module_h" + str(layer), gh)

            dp_module = nn.Dropout(dropout) # self.dropout = dropout probability.
            self.add_module("dp_module_" + str(layer), dp_module)

            # Append a tuple of 6 modules to the list.
            self.layer_params.append((dp_module, zx, zh, rx, rh, gx, gh))

        # Define the last - output module
        out_module = nn.Linear(self.h_dim, self.out_dim)
        self.add_module("out_module", out_module)
        self.layer_params.append(out_module)
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor=None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                # print ("state zeroed")
                layer_states.append(torch.zeros(batch_size, self.h_dim, device=input.device))
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None



        # TODO: Implement the model's forward pass.
        # You'll need to go layer-by-layer from bottom to top (see diagram).
        # Tip: You can use torch.stack() to combine multiple tensors into a
        # single tensor in a differentiable manner.
        # ====== YOUR CODE: ======

        # TODO: What is it for?
        layer_states.append(torch.zeros(batch_size, self.h_dim, device=input.device))
        layer_output = torch.zeros(batch_size, seq_len, self.out_dim)

        # print(self.layer_params)

        # Feeding letters. i.e.
        # First : feed all first letters in sequences.
        # Then  : feed all second letters in sequences.
        # etc.

        # This loop iterates over time: i.e. :
        #    ____     ____     ____    ____
        #-> |    |-> |    |-> |    |->|    |->
        #   |____|   |____|   |____|  |____|
        #
        for t in range(seq_len):

            # Select the current input letters.
            x = layer_input[:, t, :]

            # This loop iterates over model layers.
            # i.e. it takes the inputs to upper layer.
            for layer in range(self.n_layers):
                #print ("x input size=", x.size())
                # Get all linear modules of this layers:
                dropout, z_module_x, z_module_h_b, r_module_x, r_module_h_b, g_module_x, g_module_h_b = self.layer_params[layer]

                h = layer_states[layer]

                #print ("x,h sizes=", x.size(), h.size(), " of types: ", type(x),type(h))
                #print ("x=", x)
                z = torch.sigmoid(z_module_x(x) + z_module_h_b(h))
                r = torch.sigmoid(r_module_x(x) + r_module_h_b(h))
                g = torch.tanh(g_module_x(x) + g_module_h_b(torch.mul(r, h)))
                h = torch.mul(z, h) + torch.mul(1 - z, g)

                # current input (x) is prev output (dp(h)).
                x = dropout(h)

                layer_states[layer + 1] = h

            # Calc final output
            output_module = self.layer_params[self.n_layers]
            layer_output[:, t, :] = output_module(x)

        # Save hidden_state (last layer_states)
        hidden_state = torch.transpose(torch.stack(layer_states[0:-1]), 0, 1)

        # ========================
        return layer_output, hidden_state

