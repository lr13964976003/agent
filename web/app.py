from flask import Flask, request, jsonify, render_template, send_file
import subprocess
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
GENERATED_DIR = os.path.join(BASE_DIR, "generated")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/list_environments", methods=["GET"])
def list_environments():
    envs = []
    for name in os.listdir(SRC_DIR):
        env_path = os.path.join(SRC_DIR, name)
        if os.path.isdir(env_path) and os.path.exists(os.path.join(env_path, "demo.py")):
            envs.append(name)
    return jsonify({"environments": envs})

@app.route("/list_prompts", methods=["GET"])
def list_prompts():
    env = request.args.get("env")
    env_path = os.path.join(SRC_DIR, env, "prompts")
    if not os.path.exists(env_path):
        return jsonify({"error": "Invalid environment"}), 400

    files = [f for f in os.listdir(env_path) if os.path.isfile(os.path.join(env_path, f))]
    return jsonify({"prompts": files})

@app.route("/get_prompt", methods=["GET"])
def get_prompt():
    env = request.args.get("env")
    fname = request.args.get("filename")
    fpath = os.path.join(SRC_DIR, env, "prompts", fname)

    if not os.path.exists(fpath):
        return jsonify({"error": "Prompt not found"}), 404

    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    return jsonify({"filename": fname, "content": content})

@app.route("/generate", methods=["POST"])
def generate_docs():
    data = request.json
    arxiv_id = data.get("arxiv_id")
    env = data.get("env")
    prompt = data.get("prompt")

    if not arxiv_id or not env:
        return jsonify({"error": "Missing parameters"}), 400

    env_path = os.path.join(SRC_DIR, env)
    demo_path = os.path.join(env_path, "demo.py")
    if not os.path.exists(demo_path):
        return jsonify({"error": f"Environment {env} not found"}), 404

    try:
        subprocess.run(
            ["python", demo_path, f"--arxiv_id={arxiv_id}", f"--prompt={prompt}"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": f"Paper to Real 成功: {arxiv_id} ({env})"})

@app.route("/list_docs", methods=["GET"])
def list_docs():
    arxiv_id = request.args.get("arxiv_id")
    folder = os.path.join(GENERATED_DIR, arxiv_id)
    if not os.path.exists(folder):
        return jsonify({"error": "No documents found"}), 404
    return jsonify({"files": os.listdir(folder)})

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
