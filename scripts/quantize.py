#!/usr/bin/env python3
"""
bge-small-zh-v1.5 INT8 动态量化脚本

依赖: pip install onnxruntime transformers

流程:
  1. 从 HuggingFace 下载 BAAI/bge-small-zh-v1.5，导出 FP32 ONNX + vocab.txt
  2. onnxruntime.quantization.quantize_dynamic 动态量化 (FP32→INT8)
  3. 验证 FP32 vs INT8 余弦相似度 > 0.99

动态量化只量化权重 (FP32→QInt8)，激活值推理时动态计算。
NLP/embedding 模型精度损失 <1%，无需校准数据集。
"""

import os
from pathlib import Path


def export_fp32_onnx(output_dir: str):
    from transformers import AutoTokenizer, AutoModel
    import torch

    model_name = "BAAI/bge-small-zh-v1.5"
    print(f"[1/3] 下载模型 {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    vocab_path = os.path.join(output_dir, "vocab.txt")
    tokenizer.save_vocabulary(vocab_path)
    print(f"      词表已保存: {vocab_path} ({tokenizer.vocab_size} tokens)")

    fp32_path = os.path.join(output_dir, "model_fp32.onnx")
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 32))
    dummy_attention_mask = torch.ones(1, 32, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"      FP32 ONNX: {fp32_path} ({size_mb:.1f} MB)")
    return fp32_path


def quantize_int8(fp32_path: str, output_dir: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"[2/3] 动态量化 (FP32 -> INT8) ...")

    int8_path = os.path.join(output_dir, "model_int8.onnx")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )
    size_mb = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"      INT8 ONNX: {int8_path} ({size_mb:.1f} MB)")

    import onnx
    onnx.checker.check_model(int8_path)
    print("      模型结构验证通过")
    return int8_path


def verify_inference(int8_path: str, output_dir: str):
    print("[3/3] 验证推理一致性 ...")

    import numpy as np
    import onnxruntime as ort

    fp32_path = os.path.join(output_dir, "model_fp32.onnx")
    session_fp32 = ort.InferenceSession(fp32_path)
    session_int8 = ort.InferenceSession(int8_path)

    rng = np.random.RandomState(42)
    input_ids = rng.randint(0, 21128, (1, 32)).astype(np.int64)
    input_ids[0, 0] = 101
    input_ids[0, -1] = 102
    attention_mask = np.ones((1, 32), dtype=np.int64)

    out_fp32 = session_fp32.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    out_int8 = session_int8.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    vec_fp32 = out_fp32[0][0, 0, :].flatten()
    vec_int8 = out_int8[0][0, 0, :].flatten()

    cos_sim = np.dot(vec_fp32, vec_int8) / (np.linalg.norm(vec_fp32) * np.linalg.norm(vec_int8) + 1e-8)
    status = "PASS" if cos_sim > 0.99 else "WARN"
    print(f"      余弦相似度: {cos_sim:.6f} ({status})")


def main():
    script_dir = Path(__file__).resolve().parent
    output_dir = str(script_dir.parent / "model")
    os.makedirs(output_dir, exist_ok=True)
    print(f"模型输出目录: {output_dir}\n")

    fp32_path = export_fp32_onnx(output_dir)
    print()
    int8_path = quantize_int8(fp32_path, output_dir)
    print()
    verify_inference(int8_path, output_dir)
    print()

    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"完成: FP32={fp32_size:.1f}MB -> INT8={int8_size:.1f}MB (压缩比 {fp32_size/int8_size:.1f}x)")
    print(f"  {output_dir}/model_int8.onnx  <- C++ ONNX Runtime 推理")
    print(f"  {output_dir}/vocab.txt         <- C++ GreedyTokenizer 分词")


if __name__ == "__main__":
    main()
