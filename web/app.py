import os
import re
import json
import subprocess
from flask import Flask, request, jsonify, send_file, render_template
import markdown

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
GENERATED_DIR = os.path.join(BASE_DIR, "generated_docs")

print(os.getcwd())

# ---------- 环境列表 ----------
@app.route("/list_environments", methods=["GET"])
def list_environments():
    envs = []
    for name in os.listdir(SRC_DIR):
        env_path = os.path.join(SRC_DIR, name)
        if os.path.isdir(env_path) and os.path.exists(os.path.join(env_path, "demo.py")):
            envs.append(name)
    return jsonify({"environments": envs})

# ---------- Prompt 列表 ----------
@app.route("/list_prompts", methods=["GET"])
def list_prompts():
    env = request.args.get("env")
    prompt_dir = os.path.join(SRC_DIR, env, "prompts")
    if not os.path.exists(prompt_dir):
        return jsonify({"prompts": []})
    prompts = set()
    for root, dirs, files in os.walk(prompt_dir):
        for f in files:
            match = re.match(r'(.+)_v\d+\.\w+$', f)
            if match:
                base_name = match.group(1)
                rel_path = os.path.relpath(root, prompt_dir)
                if rel_path != ".":
                    base_name = f"{rel_path}/{base_name}"
                prompts.add(base_name)
    return jsonify({"prompts": sorted(list(prompts))})

# ---------- 版本列表 ----------
@app.route("/list_versions", methods=["GET"])
def list_versions():
    env = request.args.get("env")
    prompt_base = request.args.get("prompt")
    if not env or not prompt_base:
        return jsonify({"versions": []})

    prompt_dir = os.path.join(SRC_DIR, env, "prompts")
    if "/" in prompt_base:
        folder, base_name = os.path.split(prompt_base)
        search_dir = os.path.join(prompt_dir, folder)
    else:
        base_name = prompt_base
        search_dir = prompt_dir

    versions = []
    if os.path.exists(search_dir):
        for f in os.listdir(search_dir):
            if f.startswith(base_name + "_v"):
                match = re.search(r'_v(\d+)', f)
                if match:
                    versions.append(int(match.group(1)))
    versions.sort()
    return jsonify({"versions": [f"v{v}" for v in versions]})

# ---------- 获取 Prompt ----------
@app.route("/get_prompt", methods=["GET"])
def get_prompt():
    env = request.args.get("env")
    prompt_base = request.args.get("prompt")
    version = request.args.get("version")
    if not env or not prompt_base or not version:
        return jsonify({"error": "Missing parameters"}), 400

    prompt_dir = os.path.join(SRC_DIR, env, "prompts")
    if "/" in prompt_base:
        folder, base_name = os.path.split(prompt_base)
        fpath = os.path.join(prompt_dir, folder, f"{base_name}_{version}.txt")
    else:
        base_name = prompt_base
        fpath = os.path.join(prompt_dir, f"{base_name}_{version}.txt")

    if not os.path.exists(fpath):
        return jsonify({"error": "Prompt not found"}), 404

    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    return jsonify({"filename": os.path.basename(fpath), "content": content})

# ---------- 保存新版本 ----------
@app.route("/save_prompt", methods=["POST"])
def save_prompt():
    data = request.json
    env = data.get("env")
    prompt_base = data.get("prompt")
    content = data.get("content")
    if not env or not prompt_base or not content:
        return jsonify({"error": "Missing parameters"}), 400

    prompt_dir = os.path.join(SRC_DIR, env, "prompts")
    if "/" in prompt_base:
        folder, base_name = os.path.split(prompt_base)
        target_dir = os.path.join(prompt_dir, folder)
    else:
        base_name = prompt_base
        target_dir = prompt_dir
    os.makedirs(target_dir, exist_ok=True)

    max_version = 0
    for f in os.listdir(target_dir):
        if f.startswith(base_name + "_v"):
            match = re.search(r'_v(\d+)', f)
            if match:
                max_version = max(max_version, int(match.group(1)))
    new_version = max_version + 1
    new_filename = f"{base_name}_v{new_version}.txt"
    new_path = os.path.join(target_dir, new_filename)

    with open(new_path, "w", encoding="utf-8") as f:
        f.write(content)

    return jsonify({"message": f"Prompt 保存为新版本成功: {new_filename}"})

# ---------- 生成 Paper to Real ----------
@app.route("/generate", methods=["POST"])
def generate_docs():
    data = request.json
    arxiv_id = data.get("arxiv_id")
    env = data.get("env")
    if not arxiv_id or not env:
        return jsonify({"error": "缺少参数"}), 400

    demo_path = os.path.join(SRC_DIR, env, "demo.py")
    if not os.path.exists(demo_path):
        return jsonify({"error": "Demo not found"}), 404

    prompts_dir = os.path.join(SRC_DIR, env, "prompts")
    all_prompts = {}

    for root, dirs, files in os.walk(prompts_dir):
        for f in files:
            match = re.match(r'(.+)_v(\d+)\.\w+$', f)
            if match:
                base_name, ver = match.groups()
                rel_path = os.path.relpath(root, prompts_dir)
                if rel_path != ".":
                    base_name = f"{rel_path}/{base_name}"
                ver = int(ver)
                if base_name not in all_prompts or ver > all_prompts[base_name]['version']:
                    with open(os.path.join(root, f), "r", encoding="utf-8") as pf:
                        content = pf.read()
                    all_prompts[base_name] = {"version": ver, "content": content}

    prompts_json = {k: f"v{v['version']}" for k, v in all_prompts.items()}

    try:
        subprocess.run(
            ["python", demo_path, f"--arxiv_id={arxiv_id}", "--prompts_json", json.dumps(prompts_json, ensure_ascii=False)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": f"Paper to Real 成功: {arxiv_id} ({env})"})

@app.route("/get_markdown", methods=["GET"])
def get_markdown():
    arxiv_id = request.args.get("arxiv_id")
    folder = os.path.join("./papers", arxiv_id)

    if not os.path.exists(folder):
        return jsonify({"error": "No paper found"}), 404

    md_path = os.path.join(folder,"paper.md")
    if os.path.exists(md_path) is False:
        return jsonify({"error": "No paper found"}), 404

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_content = markdown.markdown(md_text, extensions=['fenced_code', 'tables'])

    return jsonify({
        "html": html_content
        }
    )

# ---------- 文档管理 ----------
@app.route("/list_docs", methods=["GET"])
def list_docs():
    arxiv_id = request.args.get("arxiv_id")
    folder = os.path.join(GENERATED_DIR, arxiv_id)
    if not os.path.exists(folder):
        return jsonify({"error": "No documents found"}), 404

    # 获取文件及其修改时间
    files = sorted(
        os.listdir(folder),
        key=lambda f: os.path.getmtime(os.path.join(folder, f)),
        reverse=True  # 最新在前
    )
    return jsonify({"files": files})

@app.route("/get_doc", methods=["GET"])
def get_doc():
    arxiv_id = request.args.get("arxiv_id")
    filename = request.args.get("filename")
    file_path = os.path.join(GENERATED_DIR, arxiv_id, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return jsonify({"filename": filename, "content": content})

@app.route("/download_doc", methods=["GET"])
def download_doc():
    arxiv_id = request.args.get("arxiv_id")
    filename = request.args.get("filename")
    file_path = os.path.join(GENERATED_DIR, arxiv_id, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)

