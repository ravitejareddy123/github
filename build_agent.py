import autogen
import subprocess
import json
import os
from datetime import datetime
import sqlite3
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Custom LLM client
class CustomLLMClient:
    def create(self, params):
        prompt = params.get("prompt", "")
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            response_json = json.loads(response_text.split("```json\n")[-1].split("```")[0]) if "```json" in response_text else {"message": response_text}
        except:
            response_json = {"message": response_text}
        return {"choices": [{"message": {"content": json.dumps(response_json)}}]}

# LLM configuration
llm_config = {
    "functions": [
        {
            "name": "build_and_push",
            "description": "Build and push Docker image to GHCR",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ],
    "config_list": [{"model": "gpt2", "api_key": "not-needed"}],
}

# Build agent
build_agent = autogen.AssistantAgent(
    name="BuildAgent",
    llm_config=False,  # Prevents default OpenAI client creation
    system_message="Build and push Docker image to GHCR, analyze build output, and suggest mitigations for failures. Return JSON summary."
)
build_agent.llm_client = CustomLLMClient()



# Store training summary
def store_training_summary(summary):
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS training_data
                 (agent_name TEXT, timestamp TEXT, summary TEXT)''')
    c.execute('INSERT INTO training_data VALUES (?, ?, ?)',
              ('build_agent', datetime.now().isoformat(), json.dumps(summary)))
    conn.commit()
    conn.close()

# Build and push
def build_and_push(_):
    summary = {"status": "unknown", "error_count": 0, "issues": [], "mitigations": []}
    try:
        result = subprocess.run(
            ["docker", "build", "-t", f"{os.getenv('DOCKER_IMAGE')}:latest", "."],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            summary["status"] = "success"
            subprocess.run(
                ["docker", "push", f"{os.getenv('DOCKER_IMAGE')}:latest"],
                check=True
            )
        else:
            summary["status"] = "failed"
            summary["error_count"] = 1
            summary["issues"].append("Docker build failed")
            summary["mitigations"].append("Check Dockerfile or dependencies")

        # Use GPT-2 for analysis
        prompt = f"Build summary: {json.dumps(summary)}\nAnalyze and suggest mitigations. Return JSON."
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            analysis = json.loads(response_text.split("```json\n")[-1].split("```")[0]) if "```json" in response_text else {"issues": [], "mitigations": []}
            summary["issues"].extend(analysis.get("issues", []))
            summary["mitigations"].extend(analysis.get("mitigations", []))
        except:
            summary["mitigations"].append(response_text[:100])

        store_training_summary(summary)
        with open("build_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        return json.dumps(summary, indent=2)
    except Exception as e:
        summary["status"] = "failed"
        summary["error_count"] += 1
        summary["issues"].append(str(e))
        summary["mitigations"].append("Verify Docker setup and GHCR credentials")
        store_training_summary(summary)
        with open("build_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        return json.dumps(summary, indent=2)

build_agent.register_function(function_map={"build_and_push": build_and_push})
build_agent.llm_client = CustomLLMClient()


if __name__ == "__main__":
    print(build_and_push({}))
