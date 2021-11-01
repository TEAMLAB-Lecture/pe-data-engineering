import argparse

import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms

from tqdm import tqdm
import matplotlib.pyplot as plt

from network import (
    PositionalEncoding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForwardNetwork,
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
    Transformer,
)

SEED = 42
torch.manual_seed(SEED)

# Defining Model
def look_network(device: str):
    pos_encoding = PositionalEncoding(10000, 512)(torch.zeros(1, 64, 512))
    plt.pcolormesh(pos_encoding[0].numpy(), cmap="RdBu")
    plt.xlabel("Depth")
    plt.xlim((0, 512))
    plt.ylabel("Position")
    plt.colorbar()
    plt.show()

    y = torch.rand(1, 60, 512)
    out = ScaledDotProductAttention()(y, y, y)
    print("Dot Attention Shape", out[0].shape, out[1].shape)

    temp_mha = MultiHeadAttention(features=512, num_heads=8)
    out, attn = temp_mha(q=torch.rand(1, 45, 512), k=y, v=y, mask=None)
    print("Multi Attention Shape", out.shape, attn.shape)

    sample_ffn = FeedForwardNetwork(512, 2048)
    print("Feed Forward Shape", sample_ffn(torch.rand(64, 50, 512)).shape)

    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_layer(torch.rand(64, 43, 512), None)
    print(
        "Encoder Shape", sample_encoder_layer_output.shape
    )  # (batch_size, input_seq_len, d_model)

    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_layer(torch.rand(64, 50, 512), None)
    print(
        "Encoder Shape", sample_encoder_layer_output.shape
    )  # (batch_size, input_seq_len, d_model)

    sample_encoder = Encoder(
        num_layers=2,
        features=512,
        num_heads=8,
        fffeatures=2048,
        input_vocab_size=8500,
        maximum_position_encoding=10000,
    ).to(device)
    temp_input = torch.rand(64, 62).type(torch.LongTensor).to(device)
    sample_encoder_output = sample_encoder(temp_input, mask=None)
    print(
        "Encoder Shape", sample_encoder_output.shape
    )  # (batch_size, input_seq_len, d_model)

    sample_decoder = Decoder(
        num_layers=2,
        features=512,
        num_heads=8,
        fffeatures=2048,
        target_vocab_size=8500,
        maximum_position_encoding=10000,
    ).to(device)
    temp_input = torch.rand(64, 26).type(torch.LongTensor).to(device)
    output, attn = sample_decoder(
        temp_input,
        enc_output=sample_encoder_output,
        look_ahead_mask=None,
        padding_mask=None,
    )
    print("Decoder Shape", output.shape, attn["decoder_layer2_block2"].shape)


# Transformer
def look_transformer(device: str = "cpu"):
    sample_transformer = Transformer(
        num_layers=2,
        features=512,
        num_heads=8,
        fffetures=2048,
        input_vocab_size=8500,
        target_vocab_size=8000,
        pe_input=10000,
        pe_target=6000,
    ).to(device)

    temp_input = torch.rand(64, 38).type(torch.LongTensor).to(device)
    temp_target = torch.rand(64, 36).type(torch.LongTensor).to(device)

    fn_out, _ = sample_transformer(
        temp_input,
        temp_target,
        enc_padding_mask=None,
        look_ahead_mask=None,
        dec_padding_mask=None,
    )

    print(
        "Transformer Shape", fn_out.shape
    )  # (batch_size, tar_seq_len, target_vocab_size)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("PyTorch version:[%s]." % (torch.__version__))

    print("This code use [%s]." % (device))

    look_network(device)

    look_transformer(device)
