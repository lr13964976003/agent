import os
import json
import re
import arxiv
import PyPDF2


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|] ', "_", name)

def download_paper(arxiv_id: str, save_dir: str):
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(search.results())

    title = sanitize_filename(result.title)
    save_dir = os.path.join(save_dir, arxiv_id)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    pdf_path = os.path.join(save_dir, f"paper.pdf")


    result.download_pdf(filename=pdf_path)
    print(f"save paper: {pdf_path}")

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    with open(os.path.join(save_dir, f"paper.md"), "w") as file:
        file.write(text)
    print(f"save paper as markdown")


def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    save_dir = config.get("savedir", "./download")
    paper_ids = config.get("paper_id", [])

    for pid in paper_ids:
        try:
            download_paper(pid, save_dir)

        except Exception as e:
            print(f"download failed {pid}: {e}")

if __name__ == "__main__":
    main()
