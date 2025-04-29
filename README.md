# EchoLens
A lens that returns (echoes) a description

## Idea
Capture real time images to generate accurate captions and then convert text to speech to guide visually impared people.

## Major Areas:
- Architecture
- Vision Encoder
- Text Decoder
- Text to Speech
- Additional features and BOT

[TG - 4/24] - Some research papers that we can refer:
1. https://ieeexplore.ieee.org/abstract/document/10890285
2. https://arxiv.org/abs/1411.4555 -- 2015
3. https://arxiv.org/abs/1502.03044 -- 2016
4. https://aclanthology.org/P18-1238/ -- 2018
5. https://arxiv.org/abs/2201.12086 -- 2022

Possible Approach:

## 🧠 Smart Project Structure for 4 Team Members

### 🎯 Goal:
- Everyone understands core concepts.
- You explore **multiple approaches**.
- You build toward a complete, real-world system.

---

## 🔧 **Phase 1: Diverse Core Implementation**

| Member | Task | Learning Focus | Output |
|--------|------|----------------|--------|
| 👩‍💻 Member A | CNN-based Image Encoder | Learn how ConvNets extract spatial features | Feature extractor module |
| 👨‍💻 Member B | ViT-based Encoder | Explore patch embeddings and self-attention | ViT-based encoder |
| 👩‍💻 Member C | LSTM/GRU Caption Decoder | Understand sequence generation and training loop | Decoder with attention |
| 👨‍💻 Member D | Transformer Decoder | Learn positional encoding and multi-head attention | Transformer-based decoder |

> Later, you can **compare BLEU/CIDEr scores** and see which pipeline generalizes better!

---

## 🧪 Phase 2: Evaluation and Ensemble

- Define **shared dataset + evaluation notebook**
- Compare models on same inputs
- Optionally, ensemble or fuse approaches (e.g., late fusion of outputs)

---

## 🚀 Phase 3: Modular Extensions

Once you have a strong baseline model:

| Module | Owner (Can rotate roles later) | Description |
|--------|-------------------------------|-------------|
| 🎤 Text-to-Speech (TTS) | Member A | Convert captions to speech (pyttsx3, gTTS) |
| 🎥 Live Feed + Caption Overlay | Member B | Capture from webcam and display captions |
| 🤖 Robot Integration | Member C | Use Raspberry Pi/Arduino for mobility |
| 🧠 Fine-Tuning/Transfer Learning | Member D | Try BLIP-style noisy data filtering or domain-specific tuning |

---

## 🧰 Project Setup Recommendations

- Use **GitHub + Branches** (`cnn-encoder`, `vit-encoder`, etc.)
- Organize code in **modular Python packages**
  ```
  project/
  ├── data/
  ├── encoders/
  │   ├── cnn_encoder.py
  │   └── vit_encoder.py
  ├── decoders/
  ├── utils/
  ├── train/
  ├── evaluate/
  ├── live_demo/
  └── main.py
  ```
- Use **Notion, Trello, or GitHub Projects** to track tasks
- Meet weekly for **knowledge sharing + demos**

---

## 🔄 Bonus Ideas for Collaborative Learning

- Everyone writes a mini blog post or summary of what they learned after each phase
- Teach each other during team calls
- Swap modules halfway through the project to get cross-exposure
