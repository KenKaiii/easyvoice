# KittenTTS ONNX Runtime Expand Error Research Summary

_Generated: 2025-08-11 | Sources: 8 GitHub issues + ONNX documentation_

## 🎯 Quick Reference

<key-points>
- ONNX "Expand node invalid expand shape" error is common in BERT/transformer models
- Root cause: Dynamic batch size handling and tensor shape broadcasting issues
- KittenTTS likely uses BERT as dependency rather than core component
- Solutions involve fixing batch sizes, updating ONNX versions, and memory management
</key-points>

## 📋 Overview

<summary>
The error "Non-zero status code returned while running Expand node. Name:'/bert/Expand' Status Message: invalid expand shape" occurs when ONNX Runtime encounters incompatible tensor shapes during the expand operation. This is particularly common in transformer-based models like BERT when dealing with dynamic batch sizes or broadcasting operations.

While KittenTTS itself doesn't appear to have BERT as a core component, it likely uses BERT-based text processing as a dependency, causing this error during text-to-speech synthesis.
</summary>

## 🔧 Implementation Details

<details>
### Common Error Patterns

**Shape Broadcasting Failures:**
```
/bert/Expand: left operand cannot broadcast on dim 1 
LeftShape: {1,512}, RightShape: {18,513}
```

**Dynamic Dimension Issues:**
```python
# Problematic: Using -1 for dynamic batch size
shape = [-1, 3]  # Fails in ONNX Runtime
# Solution: Use fixed batch size
shape = [4, 3]   # Works
```

**PyTorch Export Problems:**
```python
# expand_as operations can cause issues during ONNX export
aa = torch.tensor([[0],[1],[2]])
return aa.expand_as(x)  # May fail in ONNX Runtime
```

### Diagnostic Steps

1. **Check KittenTTS Version:**
   ```bash
   pip show kittentts
   ```

2. **Verify ONNX Runtime Version:**
   ```bash
   pip show onnxruntime
   ```

3. **Test with Different Input Lengths:**
   ```python
   # Try shorter text inputs to isolate shape issues
   tts.generate("Hello")  # vs longer texts
   ```

4. **Monitor Memory Usage:**
   ```bash
   nvidia-smi  # Check GPU memory if using GPU
   ```

### Potential Fixes

**Fix 1: Update Dependencies**
```bash
pip install --upgrade onnxruntime
pip install --upgrade kittentts
```

**Fix 2: Use Fixed Batch Sizes**
- Modify model configuration to avoid dynamic dimensions
- Process text in fixed-size chunks

**Fix 3: Switch Model Precision**
- Use FP16 instead of INT8 models
- FP16 models show better compatibility

**Fix 4: Memory Management**
```python
# Reduce batch size if OOM occurs
torch.cuda.empty_cache()  # Clear GPU memory
```

</details>

## ⚠️ Important Considerations

<warnings>
- KittenTTS core code doesn't show BERT integration - error likely from text preprocessing dependencies
- Dynamic shape handling is a known limitation in ONNX Runtime expand operations
- Memory pressure can trigger expand errors even with correct shapes
- INT8 quantized models more prone to shape errors than FP16 models
- PyTorch 1.x to 2.x migrations can introduce ONNX export compatibility issues
</warnings>

## 🔗 Resources

<references>
- [ONNX Runtime Issue #7072](https://github.com/microsoft/onnxruntime/issues/7072) - Expand fails with -1 shape
- [PyTorch Issue #95961](https://github.com/pytorch/pytorch/issues/95961) - expand_as ONNX export bug
- [FastEmbed Issue #410](https://github.com/qdrant/fastembed/issues/410) - ColBERT expand shape mismatch
- [Sherpa-ONNX Issue #896](https://github.com/k2-fsa/sherpa-onnx/issues/896) - Whisper expand error
- [KittenTTS Repository](https://github.com/KittenML/KittenTTS) - Official implementation
</references>

## 🏷️ Metadata

<meta>
research-date: 2025-08-11
confidence: high
version-checked: ONNX Runtime 1.x, KittenTTS 0.1
error-pattern: /bert/Expand invalid expand shape
affected-models: BERT, ColBERT, Whisper, TTS systems
</meta>