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
            "name": "run_unit_tests",
            "description": "Run unit tests and summarize results",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ],
    "config_list": [{"model": "gpt2", "api_key": "not-needed"}]
    # Remove "request_timeout": 120
}


# Test agent
# After defining your CustomLLMClient and llm_config...

test_agent = autogen.AssistantAgent(
    name="TestAgent",
    llm_config=False,  # Prevents OpenAI client creation
    system_message="Run unit tests, analyze results, and suggest fixes for failures. Return JSON summary."
)

test_agent.llm_client = CustomLLMClient()  # Assign GPT-2 client manually

# Store training summary
def store_training_summary(summary):
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS training_data
                 (agent_name TEXT, timestamp TEXT, summary TEXT)''')
    c.execute('INSERT INTO training_data VALUES (?, ?, ?)',
              ('test_agent', datetime.now().isoformat(), json.dumps(summary)))
    conn.commit()
    conn.close()

# Run unit tests
def run_unit_tests(_):
    summary = {"status": "unknown", "error_count": 0, "issues": [], "mitigations": []}
    try:
        # Create test file
        with open("tests/test_app.py", "w") as f:
            f.write("""
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_hello(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Hello from Microservice!' in response.data

def test_login_get(client):
    response = client.get('/login')
    assert response.status_code == 200
    assert b'Login' in response.data

def test_login_post_success(client):
    response = client.post('/login', data={'username': 'admin', 'password': 'password'})
    assert response.status_code == 200
    assert b'Login successful!' in response.data

def test_login_post_fail(client):
    response = client.post('/login', data={'username': 'wrong', 'password': 'wrong'})
    assert response.status_code == 200
    assert b'Invalid credentials' in response.data
""")
        result = subprocess.run(
            ["pytest", "tests/test_app.py", "-v"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            summary["status"] = "success"
        else:
            summary["status"] = "failed"
            summary["error_count"] = result.stdout.count("FAILED")
            summary["issues"].append("Unit tests failed")
            summary["mitigations"].append("Check test_app.py for errors")

        # Use GPT-2 for analysis
        prompt = f"Test summary: {json.dumps(summary)}\nTest output: {result.stdout[:200]}\nAnalyze and suggest mitigations. Return JSON."
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
        with open("test_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        return json.dumps(summary, indent=2)
    except Exception as e:
        summary["status"] = "failed"
        summary["error_count"] += 1
        summary["issues"].append(str(e))
        summary["mitigations"].append("Verify pytest installation and test files")
        store_training_summary(summary)
        with open("test_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        return json.dumps(summary, indent=2)

test_agent.register_function(function_map={"run_unit_tests": run_unit_tests})
test_agent.llm_client = CustomLLMClient()

if __name__ == "__main__":
    test_agent.initiate_chat(
        test_agent,
        message="Run unit tests and provide a JSON summary."
    )
