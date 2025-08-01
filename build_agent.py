import autogen
import subprocess
import json
import os
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sqlite3

# Initialize GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Custom LLM client
class CustomLLMClient:
    def create(self, params):
        prompt = params.get("prompt", "")
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        attention_mask = inputs["attention_mask"]
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=150,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            if "```json" in response_text:
                json_part = response_text.split("```json")[-1].split("```")[0]
                response_json = json.loads(json_part)
            else:
                response_json = {"message": response_text}
        except Exception:
            response_json = {"message": response_text}
        return {"choices": [{"message": {"content": json.dumps(response_json)}}]}

# Define build agent
build_agent = autogen.AssistantAgent(
    name="BuildAgent",
    llm_config=False,
    system_message="You are a Build Agent. Execute Docker build and push commands, then return a JSON summary of the build status."
)
build_agent.llm_client = CustomLLMClient()

# Store build summary
def store_build_summary(summary):
    try:
        conn = sqlite3.connect('build_data.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS build_data
                     (timestamp TEXT, summary TEXT)''')
        c.execute('INSERT INTO build_data VALUES (?, ?)',
                  (datetime.now().isoformat(), json.dumps(summary)))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

# Build and push Docker image
def build_and_push_docker(_):
    summary = {"status": "unknown", "image": "", "issues": [], "mitigations": []}
    try:
        github_actor = os.getenv("GITHUB_ACTOR", "ravitejareddy123")
        image_name = f"ghcr.io/{github_actor}/myimage:latest"
        result = subprocess.run(
            ["docker", "build", "-t", image_name, "."],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            summary["status"] = "failed"
            summary["issues"].append(f"Docker build failed: {result.stderr}")
            summary["mitigations"].append("Check Dockerfile and build context")
            store_build_summary(summary)
            with open("build_report.json", "w") as f:
                json.dump(summary, f, indent=2)
            return json.dumps(summary, indent=2)

        result = subprocess.run(
            ["docker", "push", image_name],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            summary["status"] = "success"
            summary["image"] = image_name
        else:
            summary["status"] = "failed"
            summary["issues"].append(f"Docker push failed: {result.stderr}")
            summary["mitigations"].append("Verify GHCR credentials and network")
        store_build_summary(summary)
        with open("build_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        return json.dumps(summary, indent=2)
    except Exception as e:
        summary["status"] = "failed"
        summary["issues"].append(str(e))
        summary["mitigations"].append("Check Docker installation and permissions")
        store_build_summary(summary)
        with open("build_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        return json.dumps(summary, indent=2)

# Register functions
build_agent.register_for_execution()(build_and_push_docker)

if __name__ == "__main__":
    autogen.initiate_chats([{
        "sender": autogen.UserProxyAgent(name="UserProxy"),
        "recipient": build_agent,
        "message": "Build and push the Docker image.",
        "max_turns": 1
    }])
