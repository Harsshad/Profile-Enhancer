import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import sys
import google.generativeai as genai
import time

from dotenv import load_dotenv
import os

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    print("Please set your GEMINI_API_KEY first!")
    sys.exit(1)
genai.configure(api_key=GEMINI_API_KEY)
GITHUB_HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN and not GITHUB_TOKEN.startswith('YOUR_') else {}

# ====== GEMINI COOLDOWN AND COUNTER ======
GEMINI_MIN_INTERVAL = 60  # seconds between Gemini API calls (set to 60 for 1 min....)
last_gemini_time = 0      # timestamp of last Gemini call
gemini_call_count = 0     # counter for Gemini API calls

# ======== SCORING FUNCTIONS =========
def smart_score(
    github_repos, github_stars, github_followers, github_forks, contributions_1yr,
    top_languages, leetcode_easy, leetcode_medium, leetcode_hard,
    hackerrank_badges=0, hackerrank_skills=0
):
    score = 0

    # GitHub
    score += np.log1p(min(github_repos if pd.notnull(github_repos) else 0, 30)) * 12
    score += np.log1p(min(github_stars if pd.notnull(github_stars) else 0, 3000)) * 32
    score += np.log1p(min(github_followers if pd.notnull(github_followers) else 0, 4000)) * 42
    score += np.log1p(min(github_forks if pd.notnull(github_forks) else 0, 600)) * 8
    score += np.cbrt(min(contributions_1yr if pd.notnull(contributions_1yr) else 0, 4500)) * 28

    # LeetCode
    lc_easy = max(leetcode_easy if pd.notnull(leetcode_easy) else 0, 0)
    lc_med = max(leetcode_medium if pd.notnull(leetcode_medium) else 0, 0)
    lc_hard = max(leetcode_hard if pd.notnull(leetcode_hard) else 0, 0)
    score += (min(lc_easy, 250)/250)**0.6 * 20
    score += (min(lc_med, 150)/150)**0.8 * 56
    score += (min(lc_hard, 70)/70)**1.2 * 110

    # HackerRank
    hr_badges = max(hackerrank_badges if pd.notnull(hackerrank_badges) else 0, 0)
    score += np.sqrt(hr_badges) * 9 + (15 if hr_badges >= 12 else 0)
    hr_skills = max(hackerrank_skills if pd.notnull(hackerrank_skills) else 0, 0)
    score += np.sqrt(hr_skills) * 4

    # Language diversity
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

    # Synergy bonus for strong users across all
    if (lc_hard >= 40 and github_stars >= 1000 and hr_badges >= 10):
        score *= 1.18
    elif (lc_med >= 75 and github_stars >= 500 and hr_badges >= 6):
        score *= 1.10

    return round(score, 2)

def assign_label_custom(score):
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

# ======== DATA FETCH FUNCTIONS ========
def fetch_github_data(username):
    try:
        user_url = f'https://api.github.com/users/{username}'
        repos_url = f'https://api.github.com/users/{username}/repos?per_page=100'
        user_resp = requests.get(user_url, headers=GITHUB_HEADERS)
        if user_resp.status_code == 404:
            print(f"Error: GitHub username '{username}' not found.")
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
        except Exception:
            contributions_1yr = 0
        return {
            "github_repos": repo_count,
            "github_stars": stars,
            "github_followers": followers,
            "github_forks": forks,
            "contributions_1yr": contributions_1yr,
            "top_languages": ",".join(top_languages)
        }
    except Exception as e:
        print("GitHub fetch error:", e)
        return None

def fetch_leetcode_data(username):
    try:
        url = 'https://leetcode.com/graphql/'
        headers = {'Content-Type': 'application/json'}
        query = {
            "operationName":"getUserProfile","variables":{"username":username},
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
            print(f"Error: LeetCode username '{username}' not found.")
            return None
        data = matched_user["submitStats"]["acSubmissionNum"]
        easy = next((d["count"] for d in data if d["difficulty"]=="Easy"), 0)
        medium = next((d["count"] for d in data if d["difficulty"]=="Medium"), 0)
        hard = next((d["count"] for d in data if d["difficulty"]=="Hard"), 0)
        return {"leetcode_easy": easy, "leetcode_medium": medium, "leetcode_hard": hard}
    except Exception as e:
        print("LeetCode fetch error:", e)
        return None

def fetch_hackerrank_data(username):
    try:
        url = f'https://www.hackerrank.com/{username}'
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 404:
            print(f"Error: HackerRank username '{username}' not found.")
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        badges = soup.find_all("div", class_="hacker-badge")
        badge_count = len(badges)
        skill_sect = soup.find_all("div", class_="profile-skill")
        skill_count = len(skill_sect)
        return {"hackerrank_badges": badge_count, "hackerrank_skills": skill_count}
    except Exception as e:
        print("HackerRank fetch error:", e)
        return None

# ======= GEMINI ANALYSIS SECTION-BY-SECTION WITH COOLDOWN =======
def get_gemini_review(prompt, retries=3):
    global last_gemini_time, gemini_call_count
    now = time.time()

    # Check cooldown
    if now - last_gemini_time < GEMINI_MIN_INTERVAL:
        wait_time = int(GEMINI_MIN_INTERVAL - (now - last_gemini_time))
        print(f"\nPlease wait {wait_time} seconds before making another Gemini API request.")
        print(f"Total Gemini API requests this run: {gemini_call_count}")
        return None

    model = genai.GenerativeModel('gemini-1.5-flash')

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            last_gemini_time = time.time()
            gemini_call_count += 1
            return response.text
        except Exception as e:
            if "429" in str(e):
                wait = 2 ** attempt
                print(f"[Retry {attempt+1}] Gemini rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print("Gemini error:", e)
                return None
    print("Failed after multiple retries.")
    return None

# ===== MAIN LOGIC ======
github_username = input("Enter your GitHub username: ").strip()
leetcode_username = input("Enter your LeetCode username: ").strip()
hackerrank_username = input("Enter your HackerRank username: ").strip()

print("\nFetching GitHub data...")
github_data = fetch_github_data(github_username)
if github_data is None:
    print("Stopping due to missing or invalid GitHub user.")
    sys.exit(1)
print("GitHub features:", github_data)

print("\nFetching LeetCode data...")
leetcode_data = fetch_leetcode_data(leetcode_username)
if leetcode_data is None:
    print("Stopping due to missing or invalid LeetCode user.")
    sys.exit(1)
print("LeetCode features:", leetcode_data)

print("\nFetching HackerRank data...")
hackerrank_data = fetch_hackerrank_data(hackerrank_username)
if hackerrank_data is None:
    print("Stopping due to missing or invalid HackerRank user.")
    sys.exit(1)
print("HackerRank features:", hackerrank_data)

features = {**github_data, **leetcode_data, **hackerrank_data}

score = smart_score(
    features["github_repos"],
    features["github_stars"],
    features["github_followers"],
    features["github_forks"],
    features["contributions_1yr"],
    features["top_languages"],
    features["leetcode_easy"],
    features["leetcode_medium"],
    features["leetcode_hard"],
    features["hackerrank_badges"],
    features["hackerrank_skills"]
)

category = assign_label_custom(score)

print(f"\n====== RESULT ======")
print(f"GitHub:     {github_username}")
print(f"LeetCode:   {leetcode_username}")
print(f"HackerRank: {hackerrank_username}")
print(f"Your computed score     : {score}")
print(f"Your predicted category : {category}")

print("\nDetails:")
for k, v in features.items():
    print(f"  {k}: {v}")

result_summary = f"""
GitHub:
  Public repos:      {features['github_repos']}
  Stars:             {features['github_stars']}
  Followers:         {features['github_followers']}
  Forks:             {features['github_forks']}
  Contributions (yr):{features['contributions_1yr']}
  Languages:         {features['top_languages']}

LeetCode:
  Easy solved:   {features['leetcode_easy']}
  Medium solved: {features['leetcode_medium']}
  Hard solved:   {features['leetcode_hard']}

HackerRank:
  Badges:   {features['hackerrank_badges']}
  Skills:   {features['hackerrank_skills']}

Overall Score:  {score}
Category:       {category}
"""

gemini_prompt = f"""
You are an expert coding judge and recruiter.
Given this summary of a user's developer profiles:

{result_summary}

For each section (GitHub, LeetCode, HackerRank):

1. Give a short strengths analysis, mentioning numbers or achievements you see.
2. Score each section individually out of 10 with brief reasoning.
3. Suggest the top improvement for each platform.

Then, calculate a single overall score for the user out of 100.
- Explain your weighting or reasoning (e.g., why you weighted some sections higher or lower).
- Show the formula if possible (e.g., sum or weighted average based on activity).
- Clearly state the "**Overall Score: XX/100**" at the end.

Finish with a holistic summary and any general tips for the candidate.

Format your answer clearly with headings for each section, and a final heading for "**Overall Score**
"""

print("\n====== AI Section-by-Section Analysis ======\n")
gemini_review = get_gemini_review(gemini_prompt)
if gemini_review:
    print(gemini_review)
