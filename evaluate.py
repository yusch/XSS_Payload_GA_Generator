import re

def detect_xss_in_line(line):
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

    detected_counts = []

    # 各行に対してXSSペイロードの検出
    for pattern in xss_patterns:
        matches = re.findall(pattern, line, re.IGNORECASE)
        detected_counts.append(len(matches))

    return detected_counts

def detect_xss(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            detected_counts = []
            for line in file:
                line_detected_counts = detect_xss_in_line(line)
                if line_detected_counts:
                    detected_counts.extend(line_detected_counts)
        
        if not detected_counts:
            print("XSSペイロードは検出されませんでした。")
            return

        # 最大値、最小値、平均を計算
        max_count = max(detected_counts)
        min_count = min(detected_counts)
        average_count = sum(detected_counts) / len(detected_counts)

        print(f"最大値: {max_count}")
        print(f"最小値: {min_count}")
        print(f"平均: {average_count:.2f}")
        
    except FileNotFoundError:
        print(f"指定されたファイル '{file_path}' は存在しません。")

if __name__ == "__main__":
    file_path = "xss-cheatsheet-data.txt"  # ファイルパスを適切なものに変更してください
    detect_xss(file_path)
