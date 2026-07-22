from youtube_comment_downloader import YoutubeCommentDownloader
from collections import defaultdict
from datetime import datetime, timezone
import json
import os

URLS = [
"https://vm.tiktok.com/ZNRTnW7Ub/",
"https://vm.tiktok.com/ZNRTnGNA1/",
"https://vm.tiktok.com/ZNRTn3GRU/",
"https://vm.tiktok.com/ZNRTnwxrn/",
"https://vm.tiktok.com/ZNRTnsn8u/",
"https://vm.tiktok.com/ZNRTn3kJu/",
"https://vm.tiktok.com/ZNRTn3wXT/",
"https://vm.tiktok.com/ZNRTn7VcR/",
"https://vm.tiktok.com/ZNRTnQx3R/",
"https://vm.tiktok.com/ZNRTnvVBL/",
"https://vm.tiktok.com/ZNRTnCH4J/",
"https://vm.tiktok.com/ZNRTn36UV/",
"https://vm.tiktok.com/ZNRTnVU8o/",
]

OUTPUT_FILE = "youtube_comments.json"


def main() -> None:
    downloader = YoutubeCommentDownloader()
    grouped: defaultdict[str, list] = defaultdict(list)
    total = 0

    for i, url in enumerate(URLS, 1):
        print(f"[{i}/{len(URLS)}] {url}")
        try:
            for comment in downloader.get_comments_from_url(url):
                text  = comment.get("text", "").strip()
                votes = comment.get("votes", 0)
                ts    = comment.get("time_parsed")

                if not text or not ts:
                    continue

                date_key = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                try:
                    like = int(votes)
                except (ValueError, TypeError):
                    like = 0

                grouped[date_key].append({"comment": text, "like": like})
                total += 1

            print(f"  -> {total} total comments so far")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    result = [
        {"date": d, "comments": v}
        for d, v in sorted(grouped.items())
    ]

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {total} comments in {len(result)} date groups -> {out_path}")


if __name__ == "__main__":
    main()
