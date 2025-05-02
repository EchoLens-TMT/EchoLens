# Computer Vision Project Idea

## EchoLens - A lens that returns (echoes) a description

## Team Members:
### Tushar Garg 12501320
### Tej Pratap 12502153
### Maneesh Ragavendra Kumar 12502234

## Problem Statement

Understanding the visual environment in real time is crucial for various assistive and autonomous systems. This project aims to build a generalized framework that can generate accurate and context-aware natural language descriptions of surroundings from live visual input. These descriptions will be converted to speech, enabling users—especially those with visual impairments—to better perceive and interact with the environment.

The goal is to develop a system that performs image-to-text captioning in real time using locally processed data. The solution will be designed to work efficiently on edge devices to ensure portability, responsiveness, and privacy. While a key motivation is to assist visually impaired individuals, the system is generalizable to domains like:

- Environmental monitoring

- Autonomous navigation

- Disaster response

- Surveillance and security

- Augmented and virtual reality

The project emphasizes a flexible and evolving approach to model development and optimization, with the aim of delivering a reliable, modular, and practical system for real-time visual scene understanding.

## Literature Survey:

1. IEEE 2024 – "Real-Time Scene Captioning for Assistive Vision Applications"
Proposes an edge-efficient real-time image captioning system tailored for visually impaired users. Combines lightweight CNNs and GRUs to balance performance and resource constraints on mobile devices.

2. Show and Tell: A Neural Image Caption Generator (Vinyals et al., 2015)
Introduced the first end-to-end deep learning model combining a CNN encoder and RNN decoder for image captioning. Demonstrated the effectiveness of training visual and language components jointly.

3. Show, Attend and Tell (Xu et al., 2016)
Added attention mechanisms to the encoder-decoder framework, allowing the model to dynamically focus on relevant image regions while generating each word, greatly improving caption relevance and detail.

4. Bottom-Up and Top-Down Attention for Image Captioning (Anderson et al., 2018)
Proposed a two-stage attention model using object detection to guide captioning. Achieved state-of-the-art results by combining low-level visual features with high-level language modeling.

5. BLIP: Bootstrapped Language Image Pretraining (2022)
Presented a flexible vision-language pretraining method that works well for both understanding and generation tasks. Uses synthetic captions filtered for quality to train robust multimodal models across tasks.

## Approach
- Design a real-time image captioning system that takes images or video frames as input and generates meaningful textual descriptions.

- Integrate a text-to-speech (TTS) component to convert these captions into spoken audio, making the system accessible to visually impaired users or useful in hands-free environments.

- Develop a custom encoder-decoder architecture, where:

- The vision encoder extracts visual features from images.

- The language decoder converts these features into natural language captions.

- Ensure the model is lightweight and optimized to run on edge devices (e.g., Raspberry Pi, Jetson Nano) without relying on cloud infrastructure.

- Keep the implementation modular to allow multiple team members to explore and test different model architectures and approaches.

- Build a scalable foundation that allows easy extension into advanced features like object tracking, environment analysis, and interactive voice feedback.

## Current Scope

- Develop and train a custom image captioning model (encoder-decoder architecture) using a dataset of image-caption pairs.

- Implement a text-to-speech (TTS) component to convert generated captions into speech.

- Run the complete pipeline on static images and real-time video input.

- Optimize and deploy the system on an edge device such as a Raspberry Pi or Jetson Nano.

- Keep the code modular and flexible to allow experimentation with different architectures by team members.

- Evaluate system performance based on caption accuracy, latency, and edge device compatibility.

## Future Scope

- Extend the system to assist visually impaired users by:

- Implementing priority-based captioning where urgent events like obstacles are notified before general descriptions.

- Detecting changes between frames (delta-based analysis) to reduce redundancy and improve real-time efficiency.

- Enhancing contextual awareness to emphasize relevant elements like moving objects, people, or road signs.

- Integrate the system with a self-following robotic cart that can carry items for users and navigate based on visual cues.

- Expand use cases to other domains like warehouse automation, surveillance, and smart AR systems.

- Incorporate reinforcement learning techniques to refine captioning based on user feedback and environment adaptability.
