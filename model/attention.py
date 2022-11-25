import torch
from torch import nn
from model.activations import ACT2FN
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

from torch.nn import DataParallel
import numpy as np


class AttentionDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.position=self.args.position #binary: use postional embedding or not

        self.drop = nn.Dropout(args.embd_pdrop)
        self.h = nn.ModuleList([Block(args.n_ctx, args, scale=True) for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
        self.mlp_f = nn.Linear(args.n_embd, args.n_embd)
        self.wpe = nn.Embedding(args.n_ctx, args.n_embd)
        self.add_cls_token = args.cls_token
        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, args.n_embd)) # add class token




        self.apply(self._init_weights)

        # self.have_z = have_z #binary
        # if self.have_z == True:
        #     self.z_placeholder = nn.Parameter(torch.Tensor(args.z_dim))
        #     self.z_placeholder.data.uniform_(-1, 1)

        # self.init_placeholder = nn.Parameter(torch.Tensor(args.n_embd))
        # self.init_placeholder.data.uniform_(-1, 1)

        print('Number of parameters: {}'.format(self._num_parameters()))



    def forward(self,
        inputs_embeds=None,
        past=None,
        attention_mask=None,
        position_ids=None,
        prop_embeds=None,
        head_mask=None,
        use_cache=True,
        output_attentions=None):
        '''
        '''

        # add cls_token
        if self.add_cls_token == True:
            expand_cls_token = self.cls_token.expand(inputs_embeds.shape[0], -1, -1)
            inputs_embeds = torch.cat((expand_cls_token, inputs_embeds), dim=1)

        # Input embeddings
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]





        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if prop_embeds is not None:
            assert inputs_embeds.size(0) == prop_embeds.size(
                0), 'Property embeddings do not match the size of the input'
            prop_embeds = prop_embeds[:, :inputs_embeds.size(1)]
        else:
            prop_embeds = torch.zeros_like(inputs_embeds)

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.float, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1]).repeat(inputs_embeds.size(0), 1)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            #shift mask for cls_token
            if self.add_cls_token == True:
                cls_token_mask = torch.ones((batch_size,1), dtype=torch.float32).to(self.args.device)
                attention_mask = torch.cat((cls_token_mask, attention_mask), dim=1)




            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        # head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # If embeddings are not given as the input, embed the provided word ids
        # position_embeds = self.wpe(position_ids)

        # Function embeddings
        # http://papers.nips.cc/paper/7181-attention-is-all-you-need
        # position_embeds = torch.zeros_like(inputs_embeds)
        # i = torch.arange(0, self.args.n_embd // 2, dtype=torch.float, device=inputs_embeds.device).unsqueeze(
        #     0).unsqueeze(0)
        # position_embeds[:, :, ::2] = torch.sin(
        #     position_ids.unsqueeze(-1) / 10000 ** (2 * i.type(torch.FloatTensor) / self.args.n_embd))
        # i = i[:, :, self.args.n_embd % 2]
        # position_embeds[:, :, 1::2] = torch.cos(
        #     position_ids.unsqueeze(-1) / 10000 ** (2 * i.type(torch.FloatTensor) / self.args.n_embd))
        hidden_states = inputs_embeds #+ position_embeds
        # hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)
        self.output_hidden_states=False

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                # head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.mlp_f(self.ln_f(hidden_states))
        # hidden_states = self.mlp_f(self.ln_f(hidden_states).view(-1, self.n_embd // 64, 64))

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):

            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count



class Attention(nn.Module):

    def __init__(self, args, nx, n_ctx, scale=True ):
        super(Attention, self).__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)

        assert n_state % args.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = args.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(args.attn_pdrop)
        self.resid_dropout = nn.Dropout(args.resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)
        # mask = self.bias[:, :, ns - nd: ns, :ns]
        # w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        bbb = w[0,0,:,:]





        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
            self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=True, output_attentions=False
    ):
        x = self.c_attn(x)  # x -> q, k, v
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)  # k=True for keys which transposes the last two dims
        value = self.split_heads(value)
        # Concat previous key and value tensors
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):

    def __init__(self, n_state, args):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = args.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[args.activation_function]
        self.dropout = nn.Dropout(args.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(torch.nn.Module):
    def __init__(self, n_ctx, args, scale=False):
        super().__init__()
        nx = args.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=args.layer_norm_epsilon)
        self.attn = Attention(args, nx, n_ctx, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=args.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, args)

    def forward(
            self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False,
    ):
        # Evaluate attention heads
        output_attn = self.attn.forward(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)
        # Residual connection 1
        x = x + a
        # FCNN
        m = self.mlp(self.ln_2(x))
        # Residual connection 2
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
