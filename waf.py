import re
import sys

def detect_xss(file_path):
    # XSSペイロードを検出する正規表現パターン
    xss_patterns = [
        r"<script[^>]*>.*?</script>",            # <script>タグ
        r"on\w+\s*=",                            # イベントハンドラ
        r"javascript\s*:",                       # JavaScriptプロトコル
        r"alert\s*\(",                           # alert() 関数
        r"eval\s*\(",                            # eval() 関数
        r"prompt\s*\(",                          # prompt() 関数
        r"confirm\s*\(",                         # confirm() 関数
    ]

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
    except FileNotFoundError:
        print(f"指定されたファイル '{file_path}' は存在しません。")
        return 0

    detected_count = 0

    # XSSペイロードの検出
    for pattern in xss_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        detected_count += len(matches)

    return detected_count