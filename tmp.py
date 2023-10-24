import re
import subprocess
import os
import locale
import codecs
import csv
import sys

def evaluate_html_syntax(html_file_path, tidy_path):
    enc = locale.getpreferredencoding()
    env_tmp = os.environ.copy()
    env_tmp['PYTHONIOENCODING'] = enc
    score = 0

    # Tidyを呼び出してHTMLファイルを評価し、結果をファイルに書き出す
    eval_html_path = f"{html_file_path}"
    with open(eval_html_path, "w", encoding="utf-8") as eval_file:
        subprocess.Popen([tidy_path, "-f", "html/html_checked_result.txt", html_file_path], stdout=eval_file, stderr=subprocess.PIPE, env=env_tmp)

    # HTMLファイルの評価結果を読み取り、警告とエラーの数を取得
    str_eval_result = ''
    with codecs.open("html/html_checked_result.txt", 'r', 'utf-8') as fin:
        str_eval_result = fin.read()

    # 正規表現を使用して警告とエラーの数を取得
    str_pattern = r'.*Tidy found ([0-9]+) warnings and ([0-9]+) errors.*$'
    obj_match = re.match(str_pattern, str_eval_result.replace('\t', '').replace('\r', '').replace('\n', ''))
    warnings = 0
    errors = 0
    if obj_match:
        warnings = int(obj_match.group(1))
        errors = int(obj_match.group(2))
        warnings = warnings * -1.0
        errors = errors * -0.1
        score = 3 + warnings + errors
    return score

def detect_xss(file_path,score):
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
        print(detected_count)
    score = score - detected_count
    return score

def main():
    num_html_files = 426  # この数を必要なファイルの数に合わせて変更してください
    tidy_path = "tidy"  # Tidy実行ファイルのパスを指定してください

    # パスを格納するリストを初期化
    html_file_paths = []

    # forループでHTMLファイルのパスを生成しリストに格納
    for i in range(1, num_html_files + 1):
        html_file_path = f"xss_cheatsheet_{i}.html"
        html_file_paths.append(html_file_path)

    for html_path in html_file_paths:
        score = evaluate_html_syntax(html_path, tidy_path)
        detect_xss(html_path,score)
        # print(f"ファイル '{html_path}' の評価結果:")
        # print(f"警告の数: {warnings}")
        # print(f"エラーの数: {errors}")
        # print()
        csv_filename = "html_evaluation_results.csv"
        with open(csv_filename, mode='a', newline='') as csv_file:
            fieldnames = ['score']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # writer.writeheader()
            writer.writerow({'score': score})
        
        print(f"警告とエラーの結果を '{csv_filename}' に出力しました。")

if __name__ == "__main__":
    main()
