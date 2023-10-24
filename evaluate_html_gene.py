# 入力ファイルのファイル名
input_filename = "xss-cheatsheet-data.txt"

# 入力ファイルを開いて各行を処理
with open(input_filename, "r") as input_file:
    for i, line in enumerate(input_file, start=1):
        # HTMLファイル名を生成
        output_filename = f"xss_cheatsheet_{i}.html"
        
        # 各行の文字列を取得
        xss_string = line.strip()
        
        # HTMLファイルを生成して書き込む
        with open(output_filename, "w") as output_file:
            output_file.write(f"{xss_string}\n")
        
        print(f"{output_filename} が生成されました。")
