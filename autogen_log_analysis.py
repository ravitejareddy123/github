import autogen
import pandas as pd
import subprocess
import json
import os
from datetime import datetime, timedelta
import sqlite3
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Mock log generator (for testing)
def generate_mock_logs(num_logs=50):
    timestamps = [datetime.now() - timedelta(minutes=x) for x in range(num_logs)][::-1]
    log_levels = ['INFO'] * int(0.7 * num_logs) + ['WARNING'] * int(0.2 * num_logs) + ['ERROR'] * int(0.05 * num_logs)
    status_codes = [200] * int(0.8 * num_logs) + [404] * int(0.15 * num_logs) + [500] * int(0.05 * num_logs)
    response_times = [max(10, min(1000, int(x))) for x in np.random.normal(100, 50, num_logs)]
    messages = []
    for level, code in zip(log_levels, status_codes):
        if level == 'ERROR':
            messages.append(f"HTTP {code} error: {'Internal Server Error' if code == 500 else 'Not Found'}")
        elif level == 'WARNING':
            messages.append(f"High response time detected: {response_times[len(messages)]:.2f}ms")
        else:
            messages.append("Request processed successfully")
    return pd.DataFrame({
        'Timestamp': timestamps,
        'LogLevel': log_levels,
        'StatusCode': status_codes,
        'ResponseTime': response_times,
        'Message': messages
    })

# Fetch KinD logs
def fetch_kind_logs():
    try:
        result = subprocess.run(
            ["kubectl", "logs", "-l", "app=microservice", "--all-containers", "-n", "default"],
            capture_output=True, text=True
        )
        logs = []
        for line in result.stdout.splitlines():
            try:
                parts = line.split(" ", 2)
                if len(parts) >= 3:
                    timestamp, level, message = parts[0], parts[1], parts[2]
                    logs.append({
                        'Timestamp': datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S'),
                        'LogLevel': level,
                        'Message': message,
                        'StatusCode': 200 if 'success' in message.lower() else (500 if 'error' in message.lower() else 0),
                        'ResponseTime': float(message.split('ms')[0].split()[-1]) if 'ms' in message else 0
                    })
            except:
                continue
        return pd.DataFrame(logs) if logs else generate_mock_logs()
    except Exception as e:
        print(f"Error fetching KinD logs: {e}")
        return generate_mock_logs()

# Store training summary
def store_training_summary(agent_name, summary):
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS training_data
                 (agent_name TEXT, timestamp TEXT, summary TEXT)''')
    c.execute('INSERT INTO training_data VALUES (?, ?, ?)',
              (agent_name, datetime.now().isoformat(), json.dumps(summary)))
    conn.commit()
    conn.close()

# Custom LLM client
class CustomLLMClient:
    def create(self, params):
        prompt = params.get("prompt", "")
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=150,
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
            "name": "analyze_kind_logs",
            "description": "Analyze KinD pod logs and summarize issues",
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

# Define agents
log_analyst = autogen.AssistantAgent(
    name="LogAnalyst",
    llm_config=False,  # Prevents default OpenAI client creation
    system_message="Analyze KinD logs, summarize anomalies (e.g., HTTP errors, login failures), and suggest mitigations. Return JSON summary."
)
log_analyst.llm_client = CustomLLMClient()

report_generator = autogen.AssistantAgent(
    name="ReportGenerator",
    llm_config=False,  # Prevents default OpenAI client creation
    system_message="Convert Log Analyst's JSON summary into a markdown report with sections: Log Summary, Issues, Mitigations."
)
report_generator.llm_client = CustomLLMClient()

# Group chat

groupchat = autogen.GroupChat(
    agents=[log_analyst],
    messages=[],
    max_round=5,
    allow_repeat_speaker=True  # Allow same agent to speak multiple times
)
groupchat.llm_client = CustomLLMClient()


# Analyze logs
def analyze_kind_logs(_):
    logs = fetch_kind_logs()
    error_logs = logs[logs['LogLevel'] == 'ERROR']
    warning_logs = logs[logs['LogLevel'] == 'WARNING']
    error_rate = len(error_logs) / len(logs) * 100 if len(logs) > 0 else 0
    avg_response_time = logs.get('ResponseTime', pd.Series([0])).mean()

    # Check for login failures
    login_errors = logs[logs['Message'].str.contains("login failed", case=False, na=False)]
    summary = {
        "total_logs": len(logs),
        "error_count": len(error_logs),
        "warning_count": len(warning_logs),
        "login_error_count": len(login_errors),
        "error_rate": round(error_rate, 2),
        "avg_response_time_ms": round(avg_response_time, 2),
        "issues": [],
        "mitigations": []
    }

    # Use GPT-2 for analysis
    prompt = f"Logs summary: {json.dumps(summary)}\nAnalyze and suggest mitigations for errors, including login failures. Return JSON."
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
    
    if error_rate > 5:
        summary["issues"].append(f"High error rate: {error_rate:.2f}%")
        summary["mitigations"].append("Check microservice code for HTTP 500 errors")
    if len(warning_logs) > 0:
        summary["issues"].append(f"High response times in {len(warning_logs)} logs")
        summary["mitigations"].append("Optimize code or increase pod resources")
    if len(login_errors) > 0:
        summary["issues"].append(f"Login failures: {len(login_errors)}")
        summary["mitigations"].append("Verify login credentials and form handling")

    store_training_summary("log_analyst", summary)
    with open("log_analysis_report.md", "w") as f:
        f.write("# KinD Log Analysis Report\n\n")
        f.write("## Log Summary\n")
        f.write(f"- Total Logs: {summary['total_logs']}\n")
        f.write(f"- Error Count: {summary['error_count']}\n")
        f.write(f"- Warning Count: {summary['warning_count']}\n")
        f.write(f"- Login Failures: {summary['login_error_count']}\n")
        f.write(f"- Error Rate: {summary['error_rate']}%\n")
        f.write(f"- Avg Response Time: {summary['avg_response_time_ms']} ms\n")
        f.write("\n## Issues Detected\n")
        for issue in summary['issues']:
            f.write(f"- {issue}\n")
        f.write("\n## Mitigation Recommendations\n")
        for mitigation in summary['mitigations']:
            f.write(f"- {mitigation}\n")

    return json.dumps(summary, indent=2)

log_analyst.register_function(function_map={"analyze_kind_logs": analyze_kind_logs})
for agent in [log_analyst, report_generator]:
    agent.llm_client = CustomLLMClient()

def run_log_analysis():
    log_analyst.initiate_chat(
        group_chat_manager,
        message="Analyze KinD logs and provide a JSON summary."
    )

if __name__ == "__main__":
    run_log_analysis()
