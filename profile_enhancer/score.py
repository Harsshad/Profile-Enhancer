from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # NEW: Import CORSMiddleware
from core import (
    fetch_github_data, fetch_leetcode_data, fetch_hackerrank_data,
    smart_score, assign_label_custom, get_gemini_review
)

app = FastAPI()

# NEW: Configure CORS middleware
# This allows your Flutter web app (and other specified origins) to make requests
# to your FastAPI backend.
origins = [
    "http://localhost",
    "http://localhost:8080", # Common Flutter web development port
    "http://localhost:52626", # The specific origin from your error message
    # IMPORTANT: If you deploy your Flutter app to a live URL (e.g., Firebase Hosting, another Render service),
    # you MUST add that production URL here as well. Example:
    # "https://your-deployed-flutter-app.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # List of origins that are allowed to make requests
    allow_credentials=True,      # Allow cookies to be included in cross-origin requests (if you use them)
    allow_methods=["*"],         # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],         # Allow all headers in the request
)

# Pydantic model to define the expected request body structure
class Usernames(BaseModel):
    github: str
    leetcode: str
    hackerrank: str

@app.get("/")
async def read_root():
    """
    Root endpoint for the API. Returns a simple message to confirm the service is running.
    """
    return {"message": "Profile Enhancer API is running!"}

async def gather_all_profile_data(github_username: str, leetcode_username: str, hackerrank_username: str):
    """
    Asynchronously gathers data from GitHub, LeetCode, and HackerRank.
    Raises HTTPException if any data fetching fails.
    """
    # Fetch GitHub data
    github_data = await fetch_github_data(github_username)
    if github_data is None:
        raise HTTPException(status_code=404, detail=f"GitHub username '{github_username}' not found or failed to fetch data.")

    # Fetch LeetCode data
    leetcode_data = await fetch_leetcode_data(leetcode_username)
    if leetcode_data is None:
        raise HTTPException(status_code=404, detail=f"LeetCode username '{leetcode_username}' not found or failed to fetch data.")

    # Fetch HackerRank data
    hackerrank_data = await fetch_hackerrank_data(hackerrank_username)
    if hackerrank_data is None:
        raise HTTPException(status_code=404, detail=f"HackerRank username '{hackerrank_username}' not found or failed to fetch data.")

    # Combine all fetched data into a single dictionary
    return {
        **github_data,
        **leetcode_data,
        **hackerrank_data
    }

@app.post("/score")
async def score_profile(data: Usernames):
    """
    Main endpoint to calculate a user's profile score and generate an AI review.
    Expects GitHub, LeetCode, and HackerRank usernames in the request body.
    """
    try:
        # Gather all profile data
        features = await gather_all_profile_data(data.github, data.leetcode, data.hackerrank)

        # Calculate the smart score
        computed_score = smart_score(
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

        # Assign a category label
        category_label = assign_label_custom(computed_score)

        # Prepare summary for Gemini API
        result_summary = f"""
        GitHub:
          Public repos:      {features.get('github_repos', 'N/A')}
          Stars:             {features.get('github_stars', 'N/A')}
          Followers:         {features.get('github_followers', 'N/A')}
          Forks:             {features.get('github_forks', 'N/A')}
          Contributions (yr):{features.get('contributions_1yr', 'N/A')}
          Languages:         {features.get('top_languages', 'N/A')}

        LeetCode:
          Easy solved:   {features.get('leetcode_easy', 'N/A')}
          Medium solved: {features.get('leetcode_medium', 'N/A')}
          Hard solved:   {features.get('leetcode_hard', 'N/A')}

        HackerRank:
          Badges:    {features.get('hackerrank_badges', 'N/A')}
          Skills:    {features.get('hackerrank_skills', 'N/A')}

        Overall Score:  {computed_score}
        Category:        {category_label}
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

        Format your answer clearly with headings for each section, and a final heading for "**Overall Score**".
        """

        # Get AI review from Gemini
        ai_review = await get_gemini_review(gemini_prompt)

        # Return the comprehensive result
        return {
            "score": computed_score,
            "label": category_label,
            "details": features,
            "ai_review": ai_review
        }
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions for FastAPI to handle
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors and return a generic internal server error
        print(f"Internal server error: {e}") # Log the error for debugging
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

