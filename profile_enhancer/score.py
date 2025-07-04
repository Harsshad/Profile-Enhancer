from fastapi import FastAPI
from pydantic import BaseModel
from core import fetch_github_data, fetch_leetcode_data, fetch_hackerrank_data, smart_score, assign_label_custom

app = FastAPI()

class Usernames(BaseModel):
    github: str
    leetcode: str
    hackerrank: str

def gather_all(usernames: tuple[str, str, str]):
    github_username, leetcode_username, hackerrank_username = usernames

    github_data = fetch_github_data(github_username)
    if github_data is None:
        raise ValueError(f"GitHub username '{github_username}' not found or failed.")

    leetcode_data = fetch_leetcode_data(leetcode_username)
    if leetcode_data is None:
        raise ValueError(f"LeetCode username '{leetcode_username}' not found or failed.")

    hackerrank_data = fetch_hackerrank_data(hackerrank_username)
    if hackerrank_data is None:
        raise ValueError(f"HackerRank username '{hackerrank_username}' not found or failed.")

    return {
        **github_data,
        **leetcode_data,
        **hackerrank_data
    }

@app.post("/score")
async def score(data: Usernames):
    try:
        results = gather_all((data.github, data.leetcode, data.hackerrank))

        score = smart_score(
            results["github_repos"],
            results["github_stars"],
            results["github_followers"],
            results["github_forks"],
            results["contributions_1yr"],
            results["top_languages"],
            results["leetcode_easy"],
            results["leetcode_medium"],
            results["leetcode_hard"],
            results["hackerrank_badges"],
            results["hackerrank_skills"]
        )

        label = assign_label_custom(score)

        return {
            "score": score,
            "label": label,
            "details": results
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}
