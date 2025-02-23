# Movie Recommendation System 🎬

A sophisticated content-based movie recommendation system that suggests films based on plot similarity analysis. The system leverages the MPST dataset and TF-IDF vectorization for accurate content matching.

## 📊 Dataset

The system utilizes the **MPST (Movie Plot Synopses with Tags) dataset**, comprising approximately 18,000 movie plot summaries. 

### Critical Files
- `genrewise_data.pkl`: Preprocessed genre-specific movie data (**DO NOT DELETE**)
- `config.json`: System configuration parameters (**DO NOT DELETE**)

## 🚀 Setup

### Prerequisites
- Python 3.10.10
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/movie-recommendation.git
   cd movie-recommendation
   ```

2. **Set up virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # For Linux/MacOS:
   source venv/bin/activate
   # For Windows:
   venv\Scripts\activate.bat
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Running the System

Execute the recommendation engine with:

python get_similar_movies.py "your movie preference" "./config.json"



### ⚠️ Important Notes
- Preserve `config.json` and `genrewise_data.pkl` files
- For fresh data processing:
  1. Remove these files:
     - `preprocessed_data.csv`
     - `vectorizer.pkl`
     - `vectorized_data.pkl`
  2. Run the command above
  3. System will regenerate necessary files and process recommendations

## 🎯 Sample Results

### Example Query
> "I love thrilling action movies set in space, with a comedic twist."

### Output
| Movie Title | Similarity Score |
|-------------|------------------|
| The Last House on the Left | 0.2517 |
| I tre volti della paura | 0.2499 |
| Die Another Day | 0.2235 |
| The Caller | 0.1724 |
| Bedtime Stories | 0.1703 |

## 🎥 Demo Video

[![Movie Recommendation System Demo](https://img.youtube.com/vi/T0tE6xVXd7g/maxresdefault.jpg)](https://www.youtube.com/watch?v=T0tE6xVXd7g)
<p align="center"><i>👆 Click to watch the demo video</i></p>

## 💼 Professional Note
Salary Expectations: Seeking compensation aligned with current market rates, commensurate with role requirements and experience level.
