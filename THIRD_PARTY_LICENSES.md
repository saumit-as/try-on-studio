# Third-Party Licenses & Model Provenance

This document records the provenance, licenses, and cryptographic checksums of third-party models and libraries bundled with or downloaded for **Try-On Studio**.

---

## 1. MobileSAM (Faster Segment Anything Model)

- **Authors / Organization:** Chaoning Zhang et al., Kyung Hee University
- **Repository:** [https://github.com/ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- **Paper:** "Faster Segment Anything: Towards Lightweight SAM for Mobile Applications" (arXiv:2306.14289)
- **License:** Apache License 2.0 (Apache-2.0)
- **Distribution / Format:** Quantized / Exported ONNX format for zero-shot promptable segmentation

### Model Artifacts & Cryptographic Checksums

| File | Size (Bytes) | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- |
| `models/mobile_sam.encoder.onnx` | 28,157,093 | `2961df55d9092ef6a804796b02a744923797772893fa3de2ad5d9fa26685fd21` | Vision Transformer image encoder (input: 1024 long-side float32 RGB; output: `1x256x64x64` embedding) |
| `models/mobile_sam.decoder.onnx` | 16,500,272 | `22cf85e35d14182f4b4712364264c06b22edbef63f065189586f080ef4e2f325` | Lightweight prompt-guided mask decoder (inputs: embedding, box/point coords & labels, prev_mask; output: 256x256 logits & full masks) |

### License Notice (Apache License 2.0)

```
Copyright 2023 Kyung Hee University & MobileSAM Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
