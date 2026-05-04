import subprocess

MODEL_PATH = "../CS_AgentV12.gguf"
LLAMA_BIN = "./llama.cpp/build/bin/llama-cli"   # 或 ./llama-cli（看你編譯出來的名稱）

# 生成參數（你可以調）
GEN_ARGS = [
    "--temp", "0.7",
    "--top-p", "0.9",
    "--top-k", "40",
    "--ctx-size", "4096",
    "--repeat-penalty", "1.1"
]

def run_llama(prompt):
    cmd = [
        LLAMA_BIN,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", "512"
    ] + GEN_ARGS

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def main():
    print("=== LLaMA.cpp Chat === (輸入 exit 離開)\n")

    history = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # 簡單對話格式（重點：乾淨）
        history += f"User: {user_input}\nAssistant:"

        output = run_llama(history)

        # 抓最後一段回覆（避免重複輸出整段 history）
        reply = output.split("Assistant:")[-1].strip()

        print(f"Assistant: {reply}\n")

        history += f" {reply}\n"


if __name__ == "__main__":
    main()