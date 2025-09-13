from huggingface_hub import snapshot_download

# 保存目录
save_dir = "/media/chenzhipeng/Qwen"

# 7B
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir=f"{save_dir}/Qwen2.5-7B-Instruct",
    resume_download=True
)

# 32B
snapshot_download(
    repo_id="Qwen/Qwen2.5-32B-Instruct",
    local_dir=f"{save_dir}/Qwen2.5-32B-Instruct",
    resume_download=True
)

print(">>> All models downloaded successfully.")
