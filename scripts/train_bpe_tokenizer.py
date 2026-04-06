import os
from typing import BinaryIO



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


import time
from tqdm import tqdm

def iter_text_chunks_with_monitor(
    file_path: str,
    chunk_size: int = 1_000_000,  # 1MB
    log_every: int = 5,          # 每N个chunk打印一次
):
    start_time = time.time()
    bytes_processed = 0
    chunk_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        buffer = []
        buffer_size = 0

        for line in f:
            buffer.append(line)
            buffer_size += len(line)
            bytes_processed += len(line)

            if buffer_size >= chunk_size:
                yield "".join(buffer)
                buffer = []
                buffer_size = 0
                chunk_count += 1

                if chunk_count % log_every == 0:
                    log_status(
                        prefix="📘 分词器流式处理",
                        bytes_processed=bytes_processed,
                        start_time=start_time,
                    )

        if buffer:
            yield "".join(buffer)

# 当前进程所占内存
import psutil
import os

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# 日志状态函数输出（内存占用、处理数据量、处理速度）
import time
def log_status(prefix, bytes_processed, start_time):
    elapsed = time.time() - start_time  
    mb = bytes_processed / 1024 / 1024
    throughput = mb / elapsed if elapsed > 0 else 0.0 # 计算吞吐量即每秒处理的数据量
    mem = get_memory_mb() 

    print(
        f"{prefix} | "
        f"mem={mem:7.1f} MB | "
        f"data={mb:8.1f} MB | "
        f"speed={throughput:6.2f} MB/s"
    )


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC


def train_bpe_tokenizer(
    train_file: str,
    val_file: str | None = None,
    vocab_size: int = 50257,
    num_chunks: int = 8,
    output_dir: str = "./bpe_tokenizer",
):
    os.makedirs(output_dir, exist_ok=True)

    special_tokens = [
        "<|endoftext|>",
        "<|unk|>",
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
    ]

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = NFKC()

    # 设计GPT-2风格的BPE Tokenizer
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True, 
    )

    def text_iterator():
        # 训练集
        for chunk in iter_text_chunks_with_monitor(
            train_file,
            chunk_size=1_000_000,    # 每次处理大小约为1MB的原始数据，处理完就会清除数据方便继续处理
            log_every=20,
        ):
            yield chunk

        # 验证集（可选）
        if val_file is not None:
            for chunk in iter_text_chunks_with_monitor(
                val_file,
                chunk_size=1_000_000,
                log_every=10,
            ):
                yield chunk


    print("🚀 开始训练BPE Tokenizer...")
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    print("✅ BPE Tokenizer训练完成")

    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    print(f"💾 分词器已保存至{output_dir}/tokenizer.json")

    return tokenizer
if __name__ == "__main__":
    train_path = "/mnt/miaohua/lizehua/workspace/basic/data/TinyStoriesV2-GPT4-train.txt"
    val_path = "/mnt/miaohua/lizehua/workspace/basic/data/TinyStoriesV2-GPT4-valid.txt"
    tokenizer = train_bpe_tokenizer(
        train_file=train_path,
        val_file=val_path,
        vocab_size=50257,     # 词表大小（通常设为32000或50257等）
        num_chunks=16,        # 读取文件时的分块数量（建议根据内存大小调整）
        output_dir="../data/bpe_tokenizer",
    )