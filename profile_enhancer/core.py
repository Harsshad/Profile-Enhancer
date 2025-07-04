import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import google.generativeai as genai
import time
import os
import asyncio # New import for asynchronous operations
from fastapi.concurrency import run_in_threadpool # Recommended for FastAPI for synchronous I/O

from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Retrieve API keys from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}. Ensure GEMINI_API_KEY is set in Render environment.")
    # In a production FastAPI app, you might want to log this and handle it gracefully
    # rather than crashing the server on startup.

# Setup GitHub headers with the token
GITHUB_HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN and not GITHUB_TOKEN.startswith('YOUR_') else {}

# ====== GEMINI COOLDOWN AND COUNTER ======
# These variables manage rate limiting for the Gemini API to avoid hitting quotas.
GEMINI_MIN_INTERVAL = 60  # seconds between Gemini API calls (e.g., 60 seconds for 1 minute)
last_gemini_time = 0      # timestamp of last Gemini call
gemini_call_count = 0     # counter for Gemini API calls in the current session

# ======== SCORING FUNCTIONS =========
def smart_score(
    github_repos, github_stars, github_followers, github_forks, contributions_1yr,
    top_languages, leetcode_easy, leetcode_medium, leetcode_hard,
    hackerrank_badges=0, hackerrank_skills=0
):
    """
    Calculates a comprehensive 'smart score' based on various developer profile metrics.
    The scoring uses logarithmic, cubic root, and power functions to normalize contributions
    and assign weights, preventing single high metrics from dominating the score.
    """
    score = 0

    # GitHub metrics contribute significantly to the score
    score += np.log1p(min(github_repos if pd.notnull(github_repos) else 0, 30)) * 12
    score += np.log1p(min(github_stars if pd.notnull(github_stars) else 0, 3000)) * 32
    score += np.log1p(min(github_followers if pd.notnull(github_followers) else 0, 4000)) * 42
    score += np.log1p(min(github_forks if pd.notnull(github_forks) else 0, 600)) * 8
    score += np.cbrt(min(contributions_1yr if pd.notnull(contributions_1yr) else 0, 4500)) * 28

    # LeetCode problem-solving skills are weighted based on difficulty
    lc_easy = max(leetcode_easy if pd.notnull(leetcode_easy) else 0, 0)
    lc_med = max(leetcode_medium if pd.notnull(leetcode_medium) else 0, 0)
    lc_hard = max(leetcode_hard if pd.notnull(leetcode_hard) else 0, 0)
    score += (min(lc_easy, 250)/250)**0.6 * 20 # Easy problems contribute moderately
    score += (min(lc_med, 150)/150)**0.8 * 56 # Medium problems have a higher impact
    score += (min(lc_hard, 70)/70)**1.2 * 110 # Hard problems yield the most points

    # HackerRank badges and skills reflect breadth of knowledge
    hr_badges = max(hackerrank_badges if pd.notnull(hackerrank_badges) else 0, 0)
    score += np.sqrt(hr_badges) * 9 + (15 if hr_badges >= 12 else 0) # Bonus for many badges
    hr_skills = max(hackerrank_skills if pd.notnull(hackerrank_skills) else 0, 0)
    score += np.sqrt(hr_skills) * 4

    # Language diversity bonus encourages broader skill sets
    if isinstance(top_languages, str):
        languages = [l.strip() for l in top_languages.split(",") if l.strip()]
    elif isinstance(top_languages, list):
        languages = top_languages
    else:
        languages = []
    unique_langs = len(set(languages))
    if unique_langs >= 7:
        score += 35
    elif unique_langs >= 5:
        score += 22
    elif unique_langs >= 3:
        score += 11

    # Synergy bonus for well-rounded profiles
    if (lc_hard >= 40 and github_stars >= 1000 and hr_badges >= 10):
        score *= 1.18 # Significant bonus for top performers across platforms
    elif (lc_med >= 75 and github_stars >= 500 and hr_badges >= 6):
        score *= 1.10 # Moderate bonus for strong overall performance

    return round(score, 2)

def assign_label_custom(score):
    """
    Assigns a categorical label (Beginner, Average, Good, Better, Excellent)
    based on the calculated smart score.
    """
    if score < 200:
        return "Beginner"
    elif score < 260:
        return "Average"
    elif score < 340:
        return "Good"
    elif score < 440:
        return "Better"
    else:
        return "Excellent"

# ======== DATA FETCH FUNCTIONS (NOW ASYNCHRONOUSLY WRAPPED) ========
async def fetch_github_data(username):
    """
    Fetches GitHub user data including public repos, stars, followers, forks,
    1-year contributions, and top languages. Runs synchronous requests in a thread pool.
    """
    try:
        # Run the synchronous requests.get and BeautifulSoup parsing in a separate thread
        # to avoid blocking the FastAPI event loop.
        def _sync_fetch():
            user_url = f'https://api.github.com/users/{username}'
            repos_url = f'https://api.github.com/users/{username}/repos?per_page=100'

            user_resp = requests.get(user_url, headers=GITHUB_HEADERS)
            if user_resp.status_code == 404:
                print(f"GitHub fetch error: Username '{username}' not found.")
                return None
            user_resp.raise_for_status()
            user_data = user_resp.json()

            repo_count = user_data.get('public_repos', 0)
            followers = user_data.get('followers', 0)

            repos_resp = requests.get(repos_url, headers=GITHUB_HEADERS)
            repos_resp.raise_for_status()
            repos_data = repos_resp.json()

            stars = sum(repo.get('stargazers_count', 0) for repo in repos_data)
            forks = sum(repo.get('forks_count', 0) for repo in repos_data)
            top_languages = list({repo.get("language") for repo in repos_data if repo.get("language")})

            contributions_1yr = 0
            try:
                html = requests.get(f"https://github.com/{username}").text
                soup = BeautifulSoup(html, "html.parser")
                contribs_tag = soup.find('h2', {"class": "f4 text-normal mb-2"})
                if contribs_tag:
                    import re
                    m = re.search(r'([\d,]+) contributions', contribs_tag.text)
                    if m:
                        contributions_1yr = int(m.group(1).replace(",", ""))
            except Exception as e:
                print(f"Warning: Could not scrape GitHub contributions for {username}: {e}")
                contributions_1yr = 0

            return {
                "github_repos": repo_count,
                "github_stars": stars,
                "github_followers": followers,
                "github_forks": forks,
                "contributions_1yr": contributions_1yr,
                "top_languages": ",".join(top_languages)
            }

        return await run_in_threadpool(_sync_fetch)

    except requests.exceptions.RequestException as e:
        print(f"GitHub API request error for {username}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching GitHub data for {username}: {e}")
        return None

async def fetch_leetcode_data(username):
    """
    Fetches LeetCode problem-solving statistics (Easy, Medium, Hard problems solved)
    using LeetCode's GraphQL API. Runs synchronous requests in a thread pool.
    """
    try:
        def _sync_fetch():
            url = 'https://leetcode.com/graphql/'
            headers = {'Content-Type': 'application/json'}
            query = {
                "operationName":"getUserProfile",
                "variables":{"username":username},
                "query":"""
                query getUserProfile($username: String!) {
                  allQuestionsCount { difficulty count }
                  matchedUser(username: $username) {
                    problemsSolvedBeatsStats { difficulty percentage }
                    submitStats: submitStatsGlobal {
                      acSubmissionNum { difficulty count }
                    }
                  }
                }"""
            }
            resp = requests.post(url, json=query, headers=headers)
            resp.raise_for_status()
            result = resp.json()

            matched_user = result.get("data", {}).get("matchedUser")
            if not matched_user:
                print(f"LeetCode fetch error: Username '{username}' not found or no data.")
                return None

            data = matched_user["submitStats"]["acSubmissionNum"]
            easy = next((d["count"] for d in data if d["difficulty"]=="Easy"), 0)
            medium = next((d["count"] for d in data if d["difficulty"]=="Medium"), 0)
            hard = next((d["count"] for d in data if d["difficulty"]=="Hard"), 0)

            return {"leetcode_easy": easy, "leetcode_medium": medium, "leetcode_hard": hard}

        return await run_in_threadpool(_sync_fetch)

    except requests.exceptions.RequestException as e:
        print(f"LeetCode API request error for {username}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching LeetCode data for {username}: {e}")
        return None

async def fetch_hackerrank_data(username):
    """
    Fetches HackerRank badge and skill counts by scraping the user's profile page.
    Runs synchronous requests in a thread pool.
    """
    try:
        def _sync_fetch():
            url = f'https://www.hackerrank.com/{username}'
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
            if resp.status_code == 404:
                print(f"HackerRank fetch error: Username '{username}' not found.")
                return None
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            badges = soup.find_all("div", class_="hacker-badge")
            badge_count = len(badges)

            skill_sect = soup.find_all("div", class_="profile-skill")
            skill_count = len(skill_sect)

            return {"hackerrank_badges": badge_count, "hackerrank_skills": skill_count}

        return await run_in_threadpool(_sync_fetch)

    except requests.exceptions.RequestException as e:
        print(f"HackerRank API request error for {username}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching HackerRank data for {username}: {e}")
        return None

# ======= GEMINI ANALYSIS SECTION-BY-SECTION WITH COOLDOWN =======
async def get_gemini_review(prompt, retries=3):
    """
    Generates a review using the Gemini API, with built-in rate limiting and retries.
    This function is now async to fit better within FastAPI's async context.
    """
    global last_gemini_time, gemini_call_count
    now = time.time()

    # Implement a cooldown to respect API rate limits
    if now - last_gemini_time < GEMINI_MIN_INTERVAL:
        wait_time = int(GEMINI_MIN_INTERVAL - (now - last_gemini_time))
        print(f"Gemini API Cooldown: Please wait {wait_time} seconds. Total calls: {gemini_call_count}")
        return f"Please wait {wait_time} seconds before making another Gemini API request. Too many requests."

    model = genai.GenerativeModel('gemini-1.5-flash')

    for attempt in range(retries):
        try:
            response = await model.generate_content_async(prompt) # Use async version
            last_gemini_time = time.time()
            gemini_call_count += 1
            return response.text
        except Exception as e:
            error_message = str(e)
            if "429" in error_message: # Rate limit error
                wait = 2 ** attempt
                print(f"[Retry {attempt+1}] Gemini rate limit hit. Waiting {wait}s...")
                await asyncio.sleep(wait) # Use asyncio.sleep for non-blocking sleep
            else:
                print(f"Gemini API error on attempt {attempt+1}: {error_message}")
                return f"Gemini API error: {error_message}"
    print("Failed to get Gemini review after multiple retries.")
    return "Failed to generate AI review due to multiple retries."

