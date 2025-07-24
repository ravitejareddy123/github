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
            "name": "deploy_to_kind",
            "description": "Deploy microservice to KinD cluster and check pod status",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ],
    "config_list": [{"model": "gpt2", "api_key": "not-needed"}],
    "request_timeout": 120,
}

# Deploy agent
deploy_agent = autogen.AssistantAgent(
    name="DeployAgent",
    llm_config=False,  # Prevents default OpenAI client creation
    system_message="Deploy the microservice to KinD, check pod status, and suggest mitigations if deployment fails. Return JSON summary."
)
deploy_agent.llm_client = CustomLLMClient()
# Store training summary
def store_training_summary(summary):
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS training_data
                 (agent_name TEXT, timestamp TEXT, summary TEXT)''')
    c.execute('INSERT INTO training_data VALUES (?, ?, ?)',
              ('deploy_agent', datetime.now().isoformat(), json.dumps(summary)))
    conn.commit()
    conn.close()

# Deploy to KinD
def deploy_to_kind(_):
    summary = {"status": "unknown", "error_count": 0, "pod_status": "unknown", "issues": [], "mitigations": []}
    try:
        # Apply microservice deployment
        subprocess.run(["kubectl", "apply", "-f", "deployment.yaml"], check=True)
        summary["status"] = "success"
        
        # Check pod status
        pod_result = subprocess.run(
            ["kubectl", "get", "pods", "-l", "app=microservice", "-n", "default", "-o", "json"],
            capture_output=True, text=True
        )
        if pod_result.returncode == 0:
            pods = json.loads(pod_result.stdout)
            if pods["items"]:
                pod_status = pods["items"][0]["status"]["phase"]
                summary["pod_status"] = pod_status
                if pod_status != "Running":
                    summary["issues"].append(f"Pod status: {pod_status}")
                    summary["mitigations"].append("Check pod logs with 'kubectl logs' or resource limits")
            else:
                summary["issues"].append("No pods found")
                summary["mitigations"].append("Verify deployment labels and KinD cluster")

        # Use GPT-2 for analysis
        prompt = f"Deployment summary: {json.dumps(summary)}\nAnalyze and suggest mitigations. Return JSON."
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
        with open("deploy_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        return json.dumps(summary, indent=2)
    except Exception as e:
        summary["status"] = "failed"
        summary["error_count"] += 1
        summary["issues"].append(str(e))
        summary["mitigations"].append("Ensure KinD cluster is running and kubectl is configured")
        store_training_summary(summary)
        with open("deploy_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        return json.dumps(summary, indent=2)

deploy_agent.register_function(function_map={"deploy_to_kind": deploy_to_kind})
deploy_agent.llm_client = CustomLLMClient()

if __name__ == "__main__":
    deploy_agent.initiate_chat(
        deploy_agent,
        message="Deploy the microservice to KinD and provide a JSON summary."
    )
