# Fake-News-Detection-implementing-on-texting-app
A privacy-preserving misinformation detection system using metadata-based forward tracking, NLP classification (DistilBERT), and human-in-the-loop verification.

# Privacy-Preserving Fake News Detection System

An AI-powered misinformation detection system inspired by how fake news actually spreads — through 
forwarded messages. Instead of scanning message content directly (which would compromise end-to-end 
encryption), this system tracks forward counts as metadata and triggers verification only after a 
defined threshold is crossed, combining automated NLP classification with human-in-the-loop community 
verification.

## Overview

Most fake-news detection models are trained on formal news articles, but real-world misinformation 
spreads through short, informal, emoji-heavy WhatsApp-style forwards. This project identifies and 
addresses that domain mismatch through a custom data augmentation pipeline, then builds a full-stack 
system around a fine-tuned NLP model.

## System Architecture

- **Dataset & Augmentation** — Built on the LIAR2 political fact-checking dataset, augmented with a 
  dual-method pipeline (rule-based + LLM-assisted rewriting) to bridge the gap between formal and 
  informal message styles.
- **Model** — Fine-tuned DistilBERT transformer for binary misinformation classification, with 
  confidence-thresholded predictions.
- **Backend** — FastAPI server exposing a `/verify` REST endpoint for real-time inference.
- **Frontend** — React-based chat interface simulating WhatsApp-style forwarding, with live forward 
  tracking, verification badges, and community verification (thumbs up/down) fused with the AI verdict.

## Tech Stack

Python · FastAPI · React · PyTorch · HuggingFace Transformers · DistilBERT · scikit-learn

## Project Scope & Limitations

This model is trained and validated specifically for political, social, and health-related 
misinformation-style claims — the dominant categories of real-world WhatsApp forward content. It is 
not designed as a general-purpose fact-checker for arbitrary trivia or scientific claims outside 
this domain.

## Status

Fully functional end-to-end prototype: dataset pipeline, model training, backend API, and frontend 
UI all completed and integrated.

## Author

R Megha Ganesh — [LinkedIn](https://linkedin.com/in/r-megha-ganesh)
