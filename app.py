import re
import math
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "CineClaude/1.0 (Educational Movie Recommendation Project)"}

KNOWN_GENRES = [
    "science fiction", "sci-fi", "action", "comedy", "romance", "romantic", "thriller",
    "horror", "fantasy", "biography", "biographical", "sports", "racing", "mystery",
    "crime", "adventure", "superhero", "historical", "period", "family", "drama", "musical"
]

LANGUAGES = ["telugu", "tamil", "hindi", "malayalam", "kannada", "english", "korean"]

STORY_THEMES = [
    {"label": "Time Travel & Alternate Reality", "keywords": ["time", "timeline", "future", "past", "loop", "dimension", "dream", "reality", "parallel", "memory"]},
    {"label": "Revenge & Redemption", "keywords": ["revenge", "avenge", "betrayal", "murder", "redemption", "justice", "enemy", "retribution"]},
    {"label": "Heist, Spy & Undercover", "keywords": ["heist", "robbery", "spy", "espionage", "undercover", "agent", "mission", "cartel", "mafia", "smuggling", "conspiracy"]},
    {"label": "Survival & Space Exploration", "keywords": ["survival", "space", "planet", "astronaut", "expedition", "stranded", "desert", "ocean", "disaster", "apocalypse"]},
    {"label": "Motorsport & Sports Rivalry", "keywords": ["racing", "formula", "driver", "championship", "rivalry", "cricket", "coach", "team", "tournament", "athlete"]},
    {"label": "Mythology, Prophecy & Savior", "keywords": ["god", "savior", "prophecy", "mythology", "divine", "temple", "ancient", "kingdom", "curse", "warrior", "village"]},
    {"label": "Friendship, College & Coming-of-Age", "keywords": ["friendship", "friends", "college", "students", "engineering", "youth", "journey", "road trip"]},
    {"label": "Family Bonds & Sacrifice", "keywords": ["family", "father", "mother", "son", "daughter", "brothers", "marriage", "sacrifice", "inheritance"]},
    {"label": "Love, Heartbreak & Separation", "keywords": ["love", "romance", "lover", "relationship", "marriage", "letters", "heartbreak", "reunite"]}
]

USER_CLUSTERS = [
    {"sci-fi": 5.0, "thriller": 4.5, "mystery": 4.6, "time": 4.9, "space": 4.8, "action": 3.5},
    {"action": 4.9, "comedy": 4.4, "fantasy": 4.5, "mythology": 4.7, "savior": 4.8, "family": 4.2},
    {"sports": 5.0, "racing": 5.0, "biography": 4.6, "drama": 4.1, "rivalry": 4.9, "speed": 4.8},
    {"romance": 5.0, "romantic": 5.0, "family": 4.8, "drama": 4.6, "love": 4.9, "music": 4.3},
    {"crime": 4.9, "thriller": 4.8, "spy": 4.7, "undercover": 4.8, "revenge": 4.7, "heist": 4.6},
    {"comedy": 5.0, "friendship": 4.9, "college": 4.8, "adventure": 4.4, "drama": 4.0}
]

# Server-side session memory for Collaborative Filtering
USER_SEARCH_HISTORY = []


def fetch_wiki_search(query, limit=12, offset=0):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "sroffset": offset,
        "format": "json"
    }
    res = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=8)
    return res.json().get("query", {}).get("search", [])


def fetch_wiki_extracts(titles):
    if not titles:
        return []
    unique_titles = list(dict.fromkeys(titles))[:35]
    params = {
        "action": "query",
        "titles": "|".join(unique_titles),
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "format": "json"
    }
    res = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
    pages = res.json().get("query", {}).get("pages", {}).values()
    return [p for p in pages if p.get("extract") and p.get("pageid")]


def extract_metadata(text):
    lower = text.lower()
    year_match = re.search(r"\b(19[5-9]\d|20[0-2]\d)\s+(?:Indian\s+)?(?:[a-zA-Z-]+\s+){0,3}film\b", text, re.I) \
                 or re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", text[:220])
    year = int(year_match.group(1)) if year_match else None

    genres = [g for g in KNOWN_GENRES if g in lower][:2]
    detected_lang = next((lang for lang in LANGUAGES if lang in lower), "english")

    matched_themes = []
    for theme in STORY_THEMES:
        hits = sum(1 for kw in theme["keywords"] if kw in lower)
        if hits > 0:
            matched_themes.append({"label": theme["label"], "hits": hits, "keywords": theme["keywords"]})
    matched_themes.sort(key=lambda x: x["hits"], reverse=True)
    if not matched_themes:
        matched_themes = [{"label": "Character-Driven Drama", "hits": 1, "keywords": ["drama", "journey"]}]

    return {
        "year": year,
        "genres": genres,
        "detectedLang": detected_lang,
        "themes": matched_themes[:2]
    }


def build_rating_vector(text_lower):
    vec = []
    for cluster in USER_CLUSTERS:
        total, hits = 2.5, 0
        for kw, rating in cluster.items():
            if kw in text_lower:
                total += (rating - 2.5)
                hits += 1
        vec.append(min(5.0, total / math.sqrt(hits) + 2.0 if hits > 0 else 2.2))

    history_overlap = sum(1 for word in USER_SEARCH_HISTORY if word in text_lower)
    vec.append(min(5.0, 2.5 + history_overlap * 0.3))
    return np.array(vec)


def pearson_similarity(vec_a, vec_b):
    mean_a, mean_b = np.mean(vec_a), np.mean(vec_b)
    diff_a, diff_b = vec_a - mean_a, vec_b - mean_b
    denom = np.linalg.norm(diff_a) * np.linalg.norm(diff_b)
    if denom == 0:
        return 0.5
    corr = float(np.dot(diff_a, diff_b) / denom)
    return max(0.0, min(1.0, (corr + 1.0) / 2.0))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_movie():
    data = request.get_json()
    movie_name = data.get("title", "").strip()
    if not movie_name:
        return jsonify({"error": "Please enter a movie name."}), 400

    hits = fetch_wiki_search(f"{movie_name} film", limit=5)
    if not hits:
        return jsonify({"error": f'Could not find "{movie_name}" on Wikipedia.'}), 404

    pages = fetch_wiki_extracts([hits[0]["title"]])
    if not pages:
        return jsonify({"error": "No plot details found."}), 404

    target_page = pages[0]
    clean_title = re.sub(r"\s*\(.*?film.*?\)", "", target_page["title"], flags=re.I)
    meta = extract_metadata(target_page["extract"])

    # Record user interaction on backend for collaborative filtering
    words = [w for w in re.findall(r"[a-z]{4,}", target_page["extract"].lower())][:12]
    USER_SEARCH_HISTORY.extend(words)
    del USER_SEARCH_HISTORY[:-40]

    return jsonify({
        "cleanTitle": clean_title,
        "pageTitle": target_page["title"],
        "extract": target_page["extract"],
        "meta": meta
    })


@app.route("/api/recommend", methods=["POST"])
def recommend_movies():
    data = request.get_json()
    target_title = data.get("cleanTitle", "")
    target_page_title = data.get("pageTitle", "")
    target_extract = data.get("extract", "")
    meta = data.get("meta", {})
    chosen_langs = data.get("languages", ["all"])
    shown_titles = set(t.lower() for t in data.get("shownTitles", []))
    offset = int(data.get("offset", 0))

    is_all = "all" in chosen_langs
    primary_genre = meta["genres"][0] if meta.get("genres") else "drama"
    theme_kws = " OR ".join(meta["themes"][0]["keywords"][:3])
    era_year = meta.get("year") or 2020
    nearby_years = " OR ".join(str(y) for y in range(era_year - 3, era_year + 4))

    queries = []
    if is_all:
        queries = [
            f"{primary_genre} film ({theme_kws}) ({nearby_years})",
            f"{primary_genre} film ({era_year - 2} OR {era_year} OR {era_year + 2})"
        ]
    else:
        for lang in chosen_langs:
            lang_term = "(American OR British)" if lang == "english" else f'"{lang}-language"'
            queries.append(f"{lang_term} {primary_genre} film ({theme_kws}) ({nearby_years})")
            queries.append(f"{lang_term} {primary_genre} film ({era_year - 2} OR {era_year} OR {era_year + 2})")

    candidate_titles = []
    for q in queries[:6]:
        for item in fetch_wiki_search(q, limit=12, offset=offset):
            t = item["title"]
            if t.lower() != target_page_title.lower() and not t.lower().startswith("list of"):
                candidate_titles.append(t)

    candidate_pages = fetch_wiki_extracts(candidate_titles)
    valid_films = []
    for p in candidate_pages:
        lower = p["extract"].lower()
        clean_c = re.sub(r"\s*\(.*?film.*?\)", "", p["title"], flags=re.I)
        if ("film" in lower or "directed by" in lower) and clean_c.lower() != target_title.lower() and clean_c.lower() not in shown_titles:
            if is_all or any(
                ("american" in lower or "british" in lower or "english-language" in lower) if l == "english"
                else (f"{l}-language" in lower or l in lower)
                for l in chosen_langs
            ):
                valid_films.append((clean_c, p["extract"]))

    if not valid_films:
        return jsonify({"recommendations": []})

    # 1. Content-Based Filtering via Scikit-Learn TF-IDF + Cosine Similarity
    corpus = [f"{target_title} {target_extract}"] + [f"{t} {ext}" for t, ext in valid_films]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cos_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # 2. Collaborative Filtering via User-Cluster Rating Matrix
    target_rating_vec = build_rating_vector(target_extract.lower())

    results = []
    for idx, (c_title, c_extract) in enumerate(valid_films):
        c_meta = extract_metadata(c_extract)
        shared_theme = next((ct for ct in c_meta["themes"] if any(mt["label"] == ct["label"] for mt in meta["themes"])), None)
        theme_bonus = 0.25 if shared_theme else 0.0
        shared_genres = len(set(meta.get("genres", [])) & set(c_meta["genres"]))
        genre_bonus = shared_genres * 0.16

        has_years = bool(meta.get("year") and c_meta["year"])
        year_diff = abs(meta["year"] - c_meta["year"]) if has_years else 99
        era_bonus = max(0.0, (1.0 - year_diff / 7.0)) * 0.26 if has_years else 0.0

        content_score = round(min(99.0, max(48.0, 44.0 + (float(cos_sims[idx]) + theme_bonus + genre_bonus + era_bonus) * 85.0)), 1)

        c_rating_vec = build_rating_vector(c_extract.lower())
        collab_score = round(min(99.0, max(50.0, 46.0 + pearson_similarity(target_rating_vec, c_rating_vec) * 52.0)), 1)

        # 3. Hybrid Score (55% Content + 45% Collaborative)
        hybrid_score = round(content_score * 0.55 + collab_score * 0.45, 1)

        snippet = ". ".join(c_extract.split(". ")[:2]) + "."
        results.append({
            "title": c_title,
            "year": c_meta["year"] or "Film",
            "genre": f"{c_meta['detectedLang'].capitalize()} • " + (" • ".join(g.capitalize() for g in c_meta["genres"]) or "Feature Film"),
            "storyTheme": shared_theme["label"] if shared_theme else c_meta["themes"][0]["label"],
            "yearDiff": year_diff,
            "hasBothYears": has_years,
            "snippet": snippet[:140] + "..." if len(snippet) > 140 else snippet,
            "contentScore": content_score,
            "collabScore": collab_score,
            "hybridScore": hybrid_score
        })

    # Strict ±5 Year Era Cutoff
    if meta.get("year"):
        within_5 = [r for r in results if r["hasBothYears"] and r["yearDiff"] <= 5]
        within_8 = [r for r in results if r["hasBothYears"] and r["yearDiff"] <= 8]
        if len(within_5) >= 3:
            results = within_5
        elif len(within_8) >= 2:
            results = within_8

    results.sort(key=lambda x: x["hybridScore"], reverse=True)
    return jsonify({"recommendations": results[:6]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)