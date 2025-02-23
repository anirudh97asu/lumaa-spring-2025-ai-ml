# Movie Recommendation System

This project is a content-based movie recommendation system that suggests movies based on the similarity of their plot summaries to a user's query. It uses the MPST dataset and TF-IDF vectorization to compute text similarities.

## Dataset

The dataset used is the **MPST (Movie Plot Synopses with Tags) dataset**, containing approximately 18,000 movie plot summaries. The data is preprocessed and stored in the repository. Key files include:
- `genrewise_data.pkl`: Preprocessed genre-specific movie data. **Do not delete this file.**
- `config.json`: Configuration file containing paths and parameters. **Do not delete this file.**

## Setup

### Prerequisites
- Python 3.10.10
- Virtual environment (recommended)

### Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/movie-recommendation.git
   cd movie-recommendation

   python -m venv venv
source venv/bin/activate  # On Linux/MacOS
venv\Scripts\activate.bat  # On Windows

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the System

Execute the recommendation system using:


### Important Notes:
- Do not delete `config.json` and `genrewise_data.pkl` files
- To run queries on a fresh sample of data:
  1. Delete the following files:
     - preprocessed_data.csv
     - vectorizer.pkl
     - vectorized_data.pkl
  2. Run the python command above
  3. The system will create a new dataset sample, vectorizer, and then run inference


## Sample Results

   #### Example query: "I love thrilling action movies set in space, with a comedic twist."
   
   ### Sample output:
   
   #### Title: The Last House on the Left, Score:0.25171650000812723
   
   #### Title: I tre volti della paura, Score:0.24998972827400734
   
   #### Title: Die Another Day, Score:0.22354837238939895
   
   #### Title: The Caller, Score:0.17241637227037848
   
   #### Title: Bedtime Stories, Score:0.170381686209587


## Video Demo:

[![Movie Recommendation Demo](https://img.youtube.com/vi/T0tE6xVXd7g/0.jpg)](https://www.youtube.com/watch?v=T0tE6xVXd7g)


# Salary Expectations: I'm seeking compensation aligned with the current market rate for this role and my level of experience
