import torch
from torch import nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=torch.float32):
        super().__init__()
        self.in_d = in_features
        self.out_d = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = dtype))

        std = (2 / (self.in_d + self.out_d))**0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std = std*std, a = -3*std,b = 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return x @ self.weight.T
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=torch.float32):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_embeddings,embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model 
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' input tensor of shape (batch_size, sequence_length, d_model)'''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        result = x / torch.sqrt(torch.sum(torch.pow(x,2), dim = -1, keepdim = True) / self.d_model + self.eps) * self.weight
        return result.to(in_dtype)
    

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        return x * self.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self,d_model, d_ff, device, dtype):
        super().__init__()
        self.d_model = d_model 
        self.d_ff = d_ff
        self.silu = SiLU()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
    


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        assert d_k % 2 == 0, "d_k must be even"
        self.theta = theta
        self.freq = 1.0 / torch.pow(self.theta, torch.arange(0, d_k, 2, device=device, dtype=torch.float32)/d_k)

        self.position = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        
        self.freqs = torch.outer(self.position, self.freq)
        self.cos = torch.cos(self.freqs)
        self.sin = torch.sin(self.freqs) 
        # self.register_buffer

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        assert x.shape[-1] == self.d_k 
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        even_x = x[...,0::2]
        odd_x = x[...,1::2]
        even_out = even_x * cos - odd_x * sin
        odd_out = even_x *sin + odd_x * cos
        result = torch.empty_like(x)
        result[..., 0::2] = even_out
        result[..., 1::2] = odd_out
        return result
    

 
class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,Q: Float[Tensor, " ... queries d_k"],
                        K: Float[Tensor, " ... keys d_k"],
                        V: Float[Tensor, " ... values d_v"],
                        mask: Bool[Tensor, " ... queries keys"] | None = None,) -> Float[Tensor, " ... queries d_v"]:
        d_k = torch.tensor(K.size(-1),dtype=torch.float32)
        trans_K = torch.transpose(K, -2, -1) 
        score = torch.matmul(Q,trans_K) / torch.sqrt(d_k)
        if mask is not None:
            assert mask.shape[-1] == score.shape[-1] and mask.shape[-2] == score.shape[-2] ,"mask shape != attention score shape"
            # score[~mask] = float('-inf')  #这种操作不会进行广播操作
            score = score.masked_fill(~mask, float('-inf'))    # masked_fill 会自动进行广播操作
        from transformer.nn_utils import softmax
        masked_score = softmax(score, dim = -1)

        return  torch.matmul(masked_score, V)



class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_rope=False, max_seq_len = None, theta = None,  device = None):
        super().__init__()

        self.num_heads = num_heads 
        self.d_model = d_model
        self.device = device 
        self.use_rope = use_rope
        self.q_proj = Linear(d_model, d_model, device=self.device)
        self.k_proj = Linear(d_model, d_model, device=self.device)
        self.v_proj = Linear(d_model, d_model, device=self.device)
        self.output_proj = Linear(d_model, d_model, device=self.device)

        self.attn = DotProductAttention() 
        if self.use_rope:
            self.rope = RoPE(theta, d_model // self.num_heads ,max_seq_len, device = self.device)
    
    def transpose_qkv(self, x):
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)
        x = x.permute(0,2,1,3)
        # return x.reshape(-1,x.shape[2],x.shape[3])
        return x
    
    def transpose_output(self,x):
        # x = x.reshape(-1,self.num_heads,x.shape[1],x.shape[2])
        x = x.permute(0,2,1,3)
        return x.reshape(x.shape[0],x.shape[1],-1)


    def forward(self,in_features: Float[Tensor, " ... sequence_length d_in"], token_positions = None) -> Float[Tensor, " ... sequence_length d_out"]:
        
        queries = self.transpose_qkv(self.q_proj(in_features))
        keys = self.transpose_qkv(self.k_proj(in_features))
        values = self.transpose_qkv(self.v_proj(in_features))

        if self.use_rope:
            queries = self.rope(queries,token_positions)
            keys = self.rope(keys, token_positions)

        mask = torch.tril(torch.ones(queries.shape[-2],keys.shape[-2],device=self.device,dtype=torch.bool)) 
        mask = mask.unsqueeze(0).unsqueeze(0)
        output = self.attn(queries, keys, values, mask)
        output_concat = self.transpose_output(output)

        return self.output_proj(output_concat)
        

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int , use_rope=False, max_seq_len=None, theta=None, device=None, dtype = None):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model, device = device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, use_rope, max_seq_len,theta,device)
        self.ln2 = RMSNorm(d_model, device = device, dtype = dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device,dtype = dtype)
         
    def forward(self, x : Float[Tensor, " batch sequence_length d_model"],token_positions = None) -> Float[Tensor, " batch sequence_length d_model"]:
        
        x1 = x + self.attn(self.ln1(x),token_positions)
        return x1 + self.ffn(self.ln2(x1))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size:int,num_layers:int, d_model:int, num_heads:int, d_ff, use_rope, context_length:int, theta, device, dtype=torch.float32):
        super().__init__()
        self.device = device
        self.token_embeddings = Embedding(vocab_size, d_model, device = device, dtype=dtype)
         
        self.layers = nn.ModuleDict({f"{i}":TransformerBlock(d_model,
                                                    num_heads, 
                                                    d_ff, use_rope, 
                                                    max_seq_len=context_length, 
                                                    theta=theta,
                                                    device=device,
                                                    dtype=dtype) 
                                    for i in range(num_layers)})
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    def forward(self,in_indices: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        B, S = in_indices.shape
        token_positions = torch.arange(S,device = self.device).expand(B,-1)
        x = self.token_embeddings(in_indices)
        for idx in range(len(self.layers)):
            x = self.layers[str(idx)](x,token_positions)
        return self.lm_head(self.ln_final(x))



