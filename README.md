Book Recommender System
A comprehensive book recommendation system that leverages machine learning, natural language processing, and vector search to provide intelligent book recommendations based on content analysis and emotional context.

📚 Project Overview
This project analyzes a dataset of books to create an intelligent recommendation system. It performs text classification, sentiment analysis, emotion detection, and semantic search to help users discover books that match their interests and emotional preferences.

🚀 Features
Text Classification: Automatically categorizes books into Fiction/Nonfiction using zero-shot classification
Emotion Analysis: Detects emotional themes in book descriptions
Sentiment Analysis: Analyzes the sentiment of book descriptions
Vector Search: Semantic search using sentence embeddings for finding similar books
Interactive Dashboard: Gradio-based web interface for easy interaction
Data Exploration: Comprehensive data analysis and visualization
📋 Prerequisites
Python 3.8+
Virtual environment (recommended)
GPU support (optional, but recommended for faster processing)
🔧 Setup Instructions
1. Clone the Repository
2. Create Virtual Environment
3. Install Dependencies
4. Environment Configuration
Create a .env file in the project root with necessary API keys and configurations:

📊 Dataset
The project uses a books dataset with the following key features:

ISBN13
Book titles
Authors
Descriptions
Categories
Ratings
And more...
The data goes through several transformation stages:

books_cleaned.csv - Initial cleaned dataset
books_with_categories.csv - Dataset with classified categories
books_with_emotions.csv - Dataset with emotion analysis
🔬 Project Components
1. Data Exploration (data-exploration.ipynb)
Initial data analysis
Statistical summaries
Data quality assessment
Visualization of key metrics
2. Text Classification (text-classification.ipynb)
Categorizes books into simplified categories (Fiction, Nonfiction, Children's Fiction, Children's Nonfiction)
Uses Facebook's BART-large-MNLI model for zero-shot classification
Achieves classification accuracy assessment through validation
Fills missing category information
Key Process:

3. Sentiment Analysis (sentiment-analysis.ipynb)
Analyzes emotional content of book descriptions
Uses NLP models to detect emotions and sentiment
Generates emotion tags and sentiment scores
4. Vector Search (vector_search.ipynb)
Implements semantic search using sentence transformers
Creates vector embeddings of book descriptions
Enables similarity-based recommendations
5. Interactive Dashboard (gradio-dashboard.py)
User-friendly web interface
Real-time recommendations
Interactive filtering and search capabilities
🎯 Usage
Running Jupyter Notebooks
Then open any of the analysis notebooks:

data-exploration.ipynb
text-classification.ipynb
sentiment-analysis.ipynb
vector_search.ipynb
Running the Dashboard
The dashboard will launch in your default web browser, providing an interactive interface for:

Searching books
Getting recommendations
Filtering by categories and emotions
Exploring book similarities
📈 Workflow
🤖 Models Used
facebook/bart-large-mnli - Zero-shot classification
Sentence Transformers - Vector embeddings for semantic search
Emotion Detection Models - For emotional theme analysis
📝 Key Files
books_cleaned.csv: Cleaned input dataset
books_with_categories.csv: Dataset with categorized books
books_with_emotions.csv: Final dataset with emotion analysis
tagged_description.txt: Text file with processed descriptions
requirements.txt: Python dependencies
.env: Environment variables (not tracked in git)
🔍 Analysis Results
The project provides insights into:

Distribution of book categories
Emotional themes across genres
Sentiment patterns in different book types
Semantic relationships between books
🛠️ Technologies Used
pandas: Data manipulation and analysis
numpy: Numerical computing
transformers: Hugging Face transformers for NLP tasks
sentence-transformers: Semantic embeddings
gradio: Interactive web interface
tqdm: Progress bars
jupyter: Interactive development environment
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📄 License
[Specify your license here]

👤 Author
[Your name/username]

🙏 Acknowledgments
Hugging Face for pre-trained models
The open-source community for various libraries used in this project
Note: Make sure to update the .env file with your specific configuration before running the notebooks or dashboard.

Claude Sonnet 4.5 • 1x
