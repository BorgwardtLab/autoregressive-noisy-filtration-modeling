"""A transformer decoder layer.

The code in this file was inspired by https://github.com/alex-matton/causal-transformer-decoder

The repository above is licensed under the MIT license.

Copyright (c) 2025 Machine Learning and Systems Biology Research Department
Copyright (c) 2021 Alexandre Matton

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import Optional

import torch.nn as nn
from torch import Tensor


class CausalTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        if not kwargs.get("batch_first", False):
            raise ValueError("Only implemented for batch-first")
        super().__init__(*args, **kwargs)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        generating: bool = False,
    ) -> Tensor:
        """
        Args:
            see CausalTransformerDecoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: bsz x seq_len x hidden_dim
                If eval mode: embedding of last token: bsz x 1 x hidden_dim
        """

        if not generating:
            # We are just doing teacher forcing
            assert src_mask is not None  # We are assuming a causal mask
            return super().forward(
                src=src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=True,
            )

        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.

        src_last_tok = src[:, -1:, :]
        if self.norm_first:
            src_last_tok = src_last_tok + self._last_tok_sa_block(
                self.norm1(src_last_tok), self.norm1(src), src_key_padding_mask
            )
            src_last_tok = src_last_tok + self._ff_block(self.norm2(src_last_tok))
        else:
            src_last_tok = self.norm1(
                src_last_tok
                + self._last_tok_sa_block(src_last_tok, src, src_key_padding_mask)
            )
            src_last_tok = self.norm2(src_last_tok + self._ff_block(src_last_tok))

        assert src_last_tok.size(1) == 1
        return src_last_tok

    def _last_tok_sa_block(self, last_tok, src, key_padding_mask):
        x = self.self_attn(
            last_tok,
            src,
            src,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)
