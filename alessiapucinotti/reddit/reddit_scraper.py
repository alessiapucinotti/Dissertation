from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
import json
import time
import random
from collections import defaultdict
from datetime import datetime

# --- CONFIGURATION ---
QUERIES = [
    "zara price", "zara quality", "zara natural fibers",
    "zara galliano", "zara pilati", "zara narciso",
    "zara vs cos", "zara vs mango", "zara vs shein",
    "zara becoming luxury",
    "zara expensive", "zara prices going up", "zara price increase",
    "why is zara so expensive", "zara inflation"
]
SUBREDDITS = ["zara", "femalefashionadvice", "malefashionadvice", "streetwear", "fashion"]
MAX_POSTS_PER_QUERY = 5
OUTPUT_FILE = "reddit_comments.json"


def parse_score(score_text: str) -> int:
    """Convert '42 points' or '1 point' into an integer. Returns 0 if not parsable."""
    try:
        return int(score_text.strip().split()[0])
    except (ValueError, IndexError):
        return 0


def scrape_reddit():
    # date (YYYY-MM-DD) -> list of comments
    grouped: defaultdict[str, list] = defaultdict(list)

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        stealth_sync(page)

        for sub in SUBREDDITS:
            for query in QUERIES:
                print(f"Searching: '{query}' in subreddit r/{sub}...")

                search_url = (
                    f"https://old.reddit.com/r/{sub}/search"
                    f"?q={query}&restrict_sr=on&sort=relevance&t=all"
                )

                try:
                    page.goto(search_url)
                    time.sleep(random.uniform(3, 6))

                    post_links = []
                    for el in page.locator(".search-result-link a.search-title").all()[:MAX_POSTS_PER_QUERY]:
                        link = el.get_attribute("href")
                        if link:
                            post_links.append(link)

                    for link in post_links:
                        try:
                            page.goto(link)
                            time.sleep(random.uniform(3, 5))

                            # Each comment on old.reddit is a div.comment
                            comment_divs = page.locator("div.comment").all()

                            for div in comment_divs:
                                # Comment text
                                md_loc = div.locator(".md")
                                if md_loc.count() == 0:
                                    continue
                                text = md_loc.first.inner_text().strip()
                                if not text or text in ("[deleted]", "[removed]"):
                                    continue

                                # Date: datetime attribute of the <time> tag
                                time_loc = div.locator("time[datetime]")
                                if time_loc.count() == 0:
                                    continue
                                dt_str = time_loc.first.get_attribute("datetime")  # e.g. "2023-01-15T10:30:00+00:00"
                                try:
                                    date_key = datetime.fromisoformat(dt_str).strftime("%Y-%m-%d")
                                except (ValueError, TypeError):
                                    continue

                                # Upvotes: span.score (e.g. "42 points")
                                score_loc = div.locator("span.score")
                                like = 0
                                if score_loc.count() > 0:
                                    like = parse_score(score_loc.first.inner_text())

                                grouped[date_key].append({
                                    "comment": text,
                                    "like": like
                                })

                        except Exception as e:
                            print(f"Error on post {link}: {e}")
                            continue

                except Exception as e:
                    print(f"Error while searching for '{query}': {e}")
                    continue

        browser.close()

    return grouped


def save_json(grouped: dict, path: str) -> None:
    result = [
        {"date": date, "comments": comments}
        for date, comments in sorted(grouped.items())
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    total = sum(len(g["comments"]) for g in result)
    print(f"Saved {total} comments in {len(result)} date groups -> {path}")


if __name__ == "__main__":
    print("Starting Reddit scraping...")
    grouped = scrape_reddit()

    if grouped:
        save_json(grouped, OUTPUT_FILE)
    else:
        print("No comments found. Check the connection or network blocking.")
