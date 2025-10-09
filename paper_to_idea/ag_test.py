import agenta as ag
from _build_agent import *

ag.init()
variant = "paper_idea"
version = 7
app = "paper_to_idea"

config = ag.ConfigManager.get_from_registry(
        app_slug=app,
        variant_slug=variant,
        variant_version=version
        )
prompt = config["prompt"]["messages"][0]["content"]
prompt = re.sub(r'<<<.*?>>>', '', prompt, flags=re.DOTALL)
prompt = prompt.format(**inputs)
print(prompt)

