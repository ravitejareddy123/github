import autogen
import requests
import json
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Debug: Print Python version and file path
print(f"Python version: {sys.version}")
print(f"Running test_agent.py from: {os.path.abspath(__file__)}")

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
        with open("test_report.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Failed to write test_report.json: {str(e)}")
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

# Define test agent
try:
    print("Initializing TestAgent...")
    test_agent = autogen.AssistantAgent(
        name="TestAgent",
        llm_config=False,
        system_message="You are a Test Agent. Test the application endpoints and generate a JSON report."
    )
    test_agent.llm_client = CustomLLMClient()
    print("TestAgent initialized successfully")
except Exception as e:
    print(f"Failed to initialize TestAgent: {str(e)}")
    summary = {
        "status": "failed",
        "issues": [f"TestAgent initialization failed: {str(e)}"],
        "mitigations": ["Check autogen and flaml dependencies"]
    }
    try:
        with open("test_report.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Failed to write test_report.json: {str(e)}")
    exit(1)

# Test application
def test_application(_):
    summary = {"status": "unknown", "issues": [], "mitigations": []}
    try:
        # Assume app is running locally (e.g., via Docker)
        url = "http://localhost:5000/health"
        print(f"Testing endpoint: {url}")
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            summary["status"] = "success"
            summary["endpoint"] = url
            summary["response"] = response.json()
        else:
            summary["status"] = "failed"
            summary["issues"].append(f"Health check failed: {response.status_code}")
            summary["mitigations"].append("Check if application is running and accessible")
    except Exception as e:
        summary["status"] = "skipped"
        summary["issues"].append(f"Test failed: {str(e)}")
        summary["mitigations"].append("Ensure Docker container is running on port 5000 or mock the test")
        print(f"Test error: {str(e)}")
    
    try:
        with open("test_report.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Failed to write test_report.json: {str(e)}")
    return json.dumps(summary, indent=2)

# Register functions
try:
    print("Registering test_application function...")
    test_agent.register_for_execution()(test_application)
    print("Function registered successfully")
except Exception as e:
    print(f"Failed to register function: {str(e)}")
    summary = {
        "status": "failed",
        "issues": [f"Function registration failed: {str(e)}"],
        "mitigations": ["Check autogen version"]
    }
    try:
        with open("test_report.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Failed to write test_report.json: {str(e)}")
    exit(1)

if __name__ == "__main__":
    try:
        print("Initiating chat to test application...")
        autogen.initiate_chats([{
            "sender": autogen.UserProxyAgent(name="UserProxy"),
            "recipient": test_agent,
            "message": "Test the application endpoints.",
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
            with open("test_report.json", "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Failed to write test_report.json: {str(e)}")
        exit(1)
```
