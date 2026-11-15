Features
      Content-Based Filtering: Uses TF-IDF vectorization and Cosine Similarity to match movie plot summaries, genres, and story themes.
      Collaborative Filtering: Computes Pearson similarity across user-cluster preference matrices and real-time search history.
      Hybrid Ranking & Era Filtering: Combines content and collaborative scores with language selection and release-era filtering.

Tech Stack
      Backend: Python, Flask, Scikit-Learn, NumPy, Requests
      Frontend: HTML5, CSS3 (Glassmorphism UI), Vanilla JavaScript
      Data Source: Wikipedia API

Project Structure
      app.py — Flask server & hybrid ML recommendation engine
      requirements.txt — Python dependencies
      README.md — Project documentation
      templates/index.html — Glassmorphism web interface

How to Run Locally
    1.Clone or download this repository:
      git clone [https://github.com/lalamshashank/CineClaude.git](https://github.com/lalamshashank/CineClaude.git)
      cd CineClaude
    2.Install the required Python libraries:
      pip install -r requirements.txt
    3.Start the Python Flask backend server:
      python app.py
    4.Open the web app in your browser:
      Navigate to http://localhost:5000 in Chrome or Edge (keep the terminal window open while using the app).
