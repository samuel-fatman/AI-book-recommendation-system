# 📚 Book Recommender System

A comprehensive book recommendation system that leverages machine learning, natural language processing, and vector search to provide intelligent book recommendations based on content analysis and emotional context.

## 🌟 Project Overview

This project analyzes a dataset of books to create an intelligent recommendation system. It performs text classification, sentiment analysis, emotion detection, and semantic search to help users discover books that match their interests and emotional preferences.

---

## ✨ Features

- 🎯 **Text Classification**: Automatically categorizes books into Fiction/Nonfiction using zero-shot classification
- 💭 **Emotion Analysis**: Detects emotional themes in book descriptions
- 😊 **Sentiment Analysis**: Analyzes the sentiment of book descriptions
- 🔍 **Vector Search**: Semantic search using sentence embeddings for finding similar books
- 🖥️ **Interactive Dashboard**: Gradio-based web interface for easy interaction
- 📊 **Data Exploration**: Comprehensive data analysis and visualization

---

## 📋 Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- GPU support (optional, but recommended for faster processing)

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone [(https://github.com/samuel-fatman/AI-book-recommendation-system.git)]
cd "Book recommender"
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv .venv-3.11

# Activate virtual environment
# On macOS/Linux:
source .venv-3.11/bin/activate

# On Windows:
.venv-3.11\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Add your environment variables here
# Example:
# HUGGINGFACE_API_KEY=your_key_here
```

---

## 📊 Dataset

The project uses a books dataset with the following key features:
- ISBN13
- Book titles
- Authors
- Descriptions
- Categories
- Ratings

### Data Pipeline:

```
books_cleaned.csv
    ↓
text-classification.ipynb
    ↓
books_with_categories.csv
    ↓
sentiment-analysis.ipynb
    ↓
books_with_emotions.csv
    ↓
vector_search.ipynb + gradio-dashboard.py
```

---

## 🔬 Project Components

### 1. 📈 Data Exploration (`data-exploration.ipynb`)
- Initial data analysis
- Statistical summaries
- Data quality assessment
- Visualization of key metrics

### 2. 🏷️ Text Classification (`text-classification.ipynb`)

Categorizes books into simplified categories using zero-shot classification.

**Categories:**
- Fiction
- Nonfiction
- Children's Fiction
- Children's Nonfiction

**Process:**

1. **Category Mapping**: Maps existing categories to simplified versions
```python
category_mapping = {
    'Fiction': "Fiction",
    'Juvenile Fiction': "Children's Fiction",
    'Biography & Autobiography': "Nonfiction",
    'History': "Nonfiction",
    'Literary Criticism': "Nonfiction",
    'Philosophy': "Nonfiction",
    'Religion': "Nonfiction",
    'Comics & Graphic Novels': "Fiction",
    'Drama': "Fiction",
    'Juvenile Nonfiction': "Children's Nonfiction",
    'Science': "Nonfiction",
    'Poetry': "Fiction"
}
```

2. **Zero-Shot Classification**: For books with missing categories
```python
from transformers import pipeline

pipe = pipeline("zero-shot-classification", 
                model="facebook/bart-large-mnli", 
                device="mps")
```

3. **Model Validation**: Tests accuracy on 600 books (300 Fiction + 300 Nonfiction)
   - Validates prediction accuracy before applying to missing categories

4. **Missing Category Prediction**: Fills in missing categories using the trained model

**Output**: `books_with_categories.csv`

### 3. 😊 Sentiment Analysis (`sentiment-analysis.ipynb`)
- Analyzes emotional content of book descriptions
- Uses NLP models to detect emotions and sentiment
- Generates emotion tags and sentiment scores
- **Output**: `books_with_emotions.csv`

### 4. 🔍 Vector Search (`vector_search.ipynb`)
- Implements semantic search using sentence transformers
- Creates vector embeddings of book descriptions
- Enables similarity-based recommendations

### 5. 🖥️ Interactive Dashboard (`gradio-dashboard.py`)
- User-friendly web interface
- Real-time recommendations
- Interactive filtering and search capabilities

---

## 🎯 Usage

### Running Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open and run notebooks in order:
# 1. data-exploration.ipynb
# 2. text-classification.ipynb
# 3. sentiment-analysis.ipynb
# 4. vector_search.ipynb
```

### Running the Dashboard

```bash
python gradio-dashboard.py
```

The dashboard will launch in your default web browser at `http://localhost:7860`

**Features:**
- 🔍 Search books by title, author, or description
- 🎯 Get personalized recommendations
- 🏷️ Filter by categories and emotions
- 📚 Explore book similarities

---

## 🧠 Models Used

| Model | Purpose | Library |
|-------|---------|---------|
| `facebook/bart-large-mnli` | Zero-shot text classification | Hugging Face Transformers |
| Sentence Transformers | Vector embeddings for semantic search | sentence-transformers |
| Emotion Detection Models | Emotional theme analysis | transformers |

---

## 📁 Project Structure

```
Book recommender/
│
├── 📓 Notebooks
│   ├── data-exploration.ipynb          # Initial data analysis
│   ├── text-classification.ipynb       # Category classification
│   ├── sentiment-analysis.ipynb        # Emotion & sentiment analysis
│   └── vector_search.ipynb             # Semantic search implementation
│
├── 📊 Data Files
│   ├── books_cleaned.csv               # Cleaned input dataset
│   ├── books_with_categories.csv       # Dataset with categories
│   ├── books_with_emotions.csv         # Final dataset with emotions
│   └── tagged_description.txt          # Processed descriptions
│
├── 🐍 Python Files
│   └── gradio-dashboard.py             # Interactive web interface
│
├── ⚙️ Configuration
│   ├── requirements.txt                # Python dependencies
│   └── .env                           # Environment variables
│
└── 📖 Documentation
    └── README.md                       # This file
```

---

## 🔄 Workflow

```mermaid
graph LR
    A[Raw Data] --> B[Data Cleaning]
    B --> C[Category Mapping]
    C --> D[Zero-Shot Classification]
    D --> E[Validation]
    E --> F[Fill Missing Categories]
    F --> G[Emotion Analysis]
    G --> H[Vector Embeddings]
    H --> I[Interactive Dashboard]
```

---

## 📈 Model Performance

### Text Classification Accuracy
- Validated on 600 books (300 Fiction + 300 Nonfiction)
- Accuracy metrics computed in `text-classification.ipynb`
- Model: `facebook/bart-large-mnli`

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Data Processing** | pandas, numpy |
| **Machine Learning** | transformers, sentence-transformers |
| **NLP** | Hugging Face Transformers |
| **Visualization** | matplotlib, seaborn |
| **Web Interface** | gradio |
| **Development** | jupyter, tqdm |

---

## 📦 Requirements

Key dependencies (see `requirements.txt` for full list):
```
pandas
numpy
transformers
sentence-transformers
gradio
tqdm
torch
```

---

## 🚦 Quick Start

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd "Book recommender"
python3.11 -m venv .venv-3.11
source .venv-3.11/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order or launch dashboard
python gradio-dashboard.py
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

[Specify your license here - e.g., MIT License]

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- 🤗 [Hugging Face](https://huggingface.co/) for pre-trained models
- 📚 Open-source community for various libraries
- 🎓 Contributors and maintainers

---

## 📞 Support

If you encounter any issues or have questions:
1. Check existing Issues
2. Create a new issue with detailed information
3. Contact the maintainers

---

## 🗺️ Roadmap

- [ ] Add more emotion categories
- [ ] Implement collaborative filtering
- [ ] Add user ratings integration
- [ ] Deploy dashboard to cloud
- [ ] Add API endpoints
- [ ] Multi-language support

---

**⭐ If you find this project helpful, please consider giving it a star!**

---

*Last Updated: October 2025*
