# ✈ Deep Learning for Airline Sentiment Intelligence  
### A Structured Engineering Pipeline for Multi-Class Review Classification  

---

## 📌 Project Overview

This project implements a **Deep Learning–based sentiment analysis system** for airline customer feedback using the **Twitter US Airline Sentiment Dataset**.

Unlike basic classroom implementations, this repository is designed as a:

- 🔬 Reproducible system  
- 🧠 Mathematically grounded model  
- 🏗 Structured engineering pipeline  
- 📊 Experiment-driven ML project  

We emphasize:

- Clean preprocessing  
- Controlled experimentation  
- Multi-class evaluation  
- Comparative modeling  
- Practical deployment readiness  

This project demonstrates that Deep Learning is not magic — it is disciplined engineering built on mathematics.

---

## ✈ Why Airline Sentiment?

Airline reviews are:

- Short and noisy  
- Emotionally charged  
- Class-imbalanced  
- Containing sarcasm and abbreviations  
- Operationally meaningful  

Sentiment intelligence in aviation enables:

- Customer satisfaction analytics  
- Real-time complaint monitoring  
- Brand reputation tracking  
- Issue escalation automation  

This makes airline sentiment classification a **high-impact NLP problem**.

---

## 📊 Dataset

**Twitter US Airline Sentiment Dataset**  
Source:  
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

### Target Classes
- Positive  
- Neutral  
- Negative  

### Input
Raw tweet text  

### Output
Predicted sentiment label ∈ {0,1,2}

---

## 🧠 Problem Formulation

We model:

fθ(x) → {0,1,2}

Where:

- x = tokenized review  
- θ = learnable parameters  
- Output = predicted sentiment class  

This is a multi-class supervised learning problem optimized using cross-entropy loss.

---

## 🏗 System Architecture

### End-to-End Pipeline

Raw Text  
↓  
Cleaning  
↓  
Tokenization  
↓  
Vocabulary Mapping  
↓  
Embedding Layer  
↓  
Neural Network  
↓  
Logits  
↓  
Softmax  
↓  
Cross Entropy Loss  
↓  
Backpropagation  
↓  
Parameter Update  

---

## 📂 Repository Structure

sentiment_project/  
│  
├── data/  
│   └── raw_dataset.csv  
│  
├── src/  
│   ├── data_loader.py  
│   ├── preprocessing.py  
│   ├── model.py  
│   ├── train.py  
│   ├── evaluate.py  
│   ├── config.py  
│   └── utils.py  
│  
├── plots/  
│   ├── loss_curve.png  
│   ├── confusion_matrix.png  
│  
├── requirements.txt  
├── report.pdf  
└── README.md  

---

## 🔬 Model Architecture

Our baseline architecture includes:

- Learnable Embedding Layer  
- Fully Connected Neural Layers  
- Dropout Regularization  
- Final Linear Classification Layer  

### Loss Function
CrossEntropyLoss  

### Optimizer
Adam  

---

## 📈 Evaluation Metrics

We evaluate performance using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- Training & Validation Loss Curves  

Graphs are saved in:

/plots/

---

## 🆚 Model Comparison Strategy

To maintain scientific rigor, we compare multiple models:

| Model | Type | Strength |
|-------|------|----------|
| Logistic Regression | Linear | Baseline |
| LSTM | Sequential | Context-aware |
| CNN (Text) | Convolutional | Local feature detection |
| Our Model | Optimized MLP + Embedding | Efficient & balanced |
| BERT (Future) | Transformer | State-of-the-art |

---

## ⭐ Our Unique Selling Proposition (USP)

### 1️⃣ Structured Engineering Over Blind Accuracy

Instead of using large transformer models immediately, we:

- Build preprocessing from scratch  
- Control vocabulary construction  
- Analyze class imbalance  
- Monitor overfitting  
- Visualize training dynamics  

This builds real understanding.

---

### 2️⃣ Lightweight & Deployable

Transformers (e.g., BERT):

- Require large compute  
- Consume high memory  
- Are heavy for edge deployment  

Our model:

- Lightweight  
- Fast to train  
- Suitable for edge inference  
- Easy to debug  
- Easier to maintain  

Ideal for real-time airline monitoring systems.

---

### 3️⃣ Controlled Trade-Off: Accuracy vs Efficiency

We optimize for:

Accuracy + Computational Efficiency  

Not just leaderboard performance.

---

### 4️⃣ Reproducibility & Clean Codebase

The repository ensures:

- Deterministic splits  
- Organized structure  
- Saved evaluation plots  
- Clear documentation  

This improves research credibility.

---

## 📊 Expected Performance

Typical benchmark accuracies:

| Model | Accuracy Range |
|--------|---------------|
| Logistic Regression | 70–75% |
| LSTM | 75–80% |
| BERT | 80–85% |
| Our Model | 75–80% (with lower compute) |

Our goal is competitive performance with better efficiency.

---

## 🔍 Class Imbalance Handling

Airline sentiment datasets often contain:

- More negative tweets  
- Fewer positive tweets  

We evaluate:

- Per-class precision  
- Per-class recall  
- Macro F1-score  

To ensure fairness across classes.

---

## 📚 Related Research

This work draws inspiration from:

1. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification.  
2. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.  
3. Zhang et al. (2015). Character-level Convolutional Networks for Text Classification.  
4. Tang et al. (2016). Aspect-level Sentiment Classification.  

However, this project emphasizes:

- Engineering discipline  
- Reproducibility  
- Deployment readiness  

---

## 🔮 Future Goals

### Model Enhancements
- LSTM implementation  
- GRU implementation  
- CNN-based text model  
- Fine-tuned BERT comparison  

### Optimization
- Class-weighted loss  
- Focal loss  
- Attention mechanism  
- Pretrained embeddings (GloVe)  

### Deployment
- REST API integration  
- Real-time dashboard  
- Model quantization  
- Edge-device deployment  

### Research Extensions
- Sarcasm detection  
- Aspect-based sentiment analysis  
- Multi-label issue classification  
- Airline delay prediction using sentiment trends  

---

## ⚙ Installation

Python 3.8+

Install dependencies:

pip install torch torchvision numpy pandas  
pip install matplotlib scikit-learn nltk  

Optional:

pip install spacy  

---

## ▶ How to Run

### 1️⃣ Place Dataset

Put raw_dataset.csv inside:

data/

### 2️⃣ Train

python src/train.py  

### 3️⃣ Evaluate

python src/evaluate.py  

---

## 📊 Generated Outputs

After training:

- plots/loss_curve.png  
- plots/confusion_matrix.png  
- report.pdf  

---

## 🧠 Reflection Philosophy

We analyze:

- Overfitting behavior  
- Bias-variance tradeoff  
- Data preprocessing impact  
- Error patterns  
- Model misclassification trends  

Deep Learning is:

Mathematics + Clean Code + Iteration + Patience

---

## 🏁 Vision

This project is not about achieving 95% accuracy.

It is about:

- Building structured ML systems  
- Understanding neural networks deeply  
- Practicing disciplined experimentation  
- Preparing for real-world AI deployment  

---

## 👥 Contributors

K. Charan  
Ahili  
Sounabha  

---

## 📜 License

Academic Use Only.
