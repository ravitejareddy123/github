import autogen
import subprocess
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys

# Debug: Print Python version and file path
print(f"Python version: {sys.version}")
print(f"Running deploy_agent.py from: {os.path.abspath(__file__)}")

# Initialize GPT-2
try:
    print("Initializing GPT-2 tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", force_download=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", force_download=True)
    print("GPT-2 initialized successfully")
except Exception as e:
    print(f"Failed to initialize GPT-2: {str(e)}")
    summary = {
        "status": "failed",
        "issues": [f"GPT-2 initialization failed: {str(e)}"],
        "mitigations": ["Check transformers and torch dependencies, ensure network access"]
    }
    try:
        with open("deploy_report.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Failed to write deploy_report.json: {str(e)}")
    exit(1)

# Custom LLM client
class CustomLLMClient:
    def create(self, params):
        try:
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
        except Exception as e:
            print(f"LLM client error: {str(e)}")
            return {"choices": [{"message": {"content": json.dumps({"error": str(e)})}}]}

# Define deploy agent
try:
    print("Initializing DeployAgent...")
    deploy_agent = autogen.AssistantAgent(
        name="DeployAgent",
        llm_config=False,
        system_message="You are a Deploy Agent. Apply Kubernetes manifests and generate a JSON report."
    )
    deploy_agent.llm_client = CustomLLMClient()
    print("DeployAgent initialized successfully")
except Exception as e:
    print(f"Failed to initialize DeployAgent: {str(e)}")
    summary = {
        "status": "failed",
        "issues": [f"DeployAgent initialization failed: {str(e)}"],
        "mitigations": ["Check autogen and flaml dependencies"]
    }
    try:
        with open("deploy_report.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Failed to write deploy_report.json: {str(e)}")
    exit(1)

# Deploy to Kubernetes
def deploy_to_kubernetes(_):
    summary = {"status": "unknown", "issues": [], "mitigations": []}
    try:
        if not os.path.exists("deployment.yaml"):
            summary["status"] = "failed"
            summary["issues"].append("deployment.yaml not found")
            summary["mitigations"].append("Ensure deployment.yaml is in the repository root")
            try:
                with open("deploy_report.json", "w") as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                print(f"Failed to write deploy_report.json: {str(e)}")
            return json.dumps(summary, indent=2)

        print("Running kubectl apply...")
        result = subprocess.run(
            ["kubectl", "apply", "-f", "deployment.yaml"],
            capture_output=True, text=True
        )
        print(f"kubectl apply stdout: {result.stdout}")
        print(f"kubectl apply stderr: {result.stderr}")
        if result.returncode == 0:
            summary["status"] = "success"
            summary["deployment"] = "microservice"
        else:
            summary["status"] = "failed"
            summary["issues"].append(f"kubectl apply failed: {result.stderr}")
            summary["mitigations"].append("Check deployment.yaml and KinD cluster")
        try:
            with open("deploy_report.json", "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Failed to write deploy_report.json: {str(e)}")
        return json.dumps(summary, indent=2)
    except Exception as e:
        summary["status"] = "failed"
        summary["issues"].append(f"Unexpected error: {str(e)}")
        summary["mitigations"].append("Check kubectl installation and KinD cluster")
        print(f"Unexpected error: {str(e)}")
        try:
            with open("deploy_report.json", "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Failed to write deploy_report.json: {str(e)}")
        return json.dumps(summary, indent=2)

# Register functions
try:
    print("Registering deploy_to_kubernetes function...")
    deploy_agent.register_for_execution()(deploy_to_kubernetes)
    print("Function registered successfully")
except Exception as e:
    print(f"Failed to register function: {str(e)}")
    summary = {
        "status": "failed",
        "issues": [f"Function registration failed: {str(e)}"],
        "mitigations": ["Check autogen version"]
    }
    try:
        with open("deploy_report.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Failed to write deploy_report.json: {str(e)}")
    exit(1)

if __name__ == "__main__":
    try:
        print("Initiating chat to deploy application...")
        autogen.initiate_chats([{
            "sender": autogen.UserProxyAgent(name="UserProxy"),
            "recipient": deploy_agent,
            "message": "Deploy the application to Kubernetes.",
            "max_turns": 1
        }])
        print("Chat initiated successfully")
    except Exception as e:
        print(f"Chat initiation failed: {str(e)}")
        summary = {
            "status": "failed",
            "issues": [f"Chat initiation failed: {str(e)}"],
            "mitigations": ["Check autogen and dependencies"]
        }
        try:
            with open("deploy_report.json", "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Failed to write deploy_report.json: {str(e)}")
        exit(1)
```
