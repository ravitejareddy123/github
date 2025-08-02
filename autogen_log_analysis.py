import autogen
import json
import pandas as pd
import numpy as np
import markdown
import sqlite3
from datetime import datetime, timedelta
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import sys

# Debug: Print Python version and file path
print(f"Python version: {sys.version}")
print(f"Running autogen_log_analysis.py from: {os.path.abspath(__file__)}")

# Initialize GPT-2
try:
    print("Initializing GPT-2 tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", force_download=True, clean_up_tokenization_spaces=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", force_download=True)
    print("GPT-2 initialized successfully")
except Exception as e:
    print(f"Failed to initialize GPT-2: {str(e)}")
    with open("log_analysis_report.html", "w") as f:
        f.write(f"<h1>Log Analysis Report</h1><p>Error: {str(e)}</p>")
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

# Define log analyst agent
try:
    print("Initializing LogAnalyst...")
    log_analyst = autogen.AssistantAgent(
        name="LogAnalyst",
        llm_config=False,
        system_message="You are a Log Analyst. Analyze logs and generate a report in markdown format."
    )
    log_analyst.llm_client = CustomLLMClient()
    print("LogAnalyst initialized successfully")
except Exception as e:
    print(f"Failed to initialize LogAnalyst: {str(e)}")
    with open("log_analysis_report.html", "w") as f:
        f.write(f"<h1>Log Analysis Report</h1><p>Error: {str(e)}</p>")
    exit(1)

# Generate mock logs
def generate_mock_logs():
    try:
        start_time = datetime.now() - timedelta(days=1)
        logs = []
        for i in range(100):
            timestamp = (start_time + timedelta(minutes=i)).isoformat()
            response_time = np.random.exponential(0.1)
            logs.append({"timestamp": timestamp, "response_time": response_time, "status": "success" if np.random.rand() > 0.1 else "failed"})
        return pd.DataFrame(logs)
    except Exception as e:
        print(f"Failed to generate mock logs: {str(e)}")
        return pd.DataFrame()

# Analyze logs
def analyze_logs(_):
    try:
        df = generate_mock_logs()
        if df.empty:
            summary = {"status": "failed", "issues": ["No logs generated"], "mitigations": ["Check log generation logic"]}
            with open("log_analysis_report.html", "w") as f:
                f.write("<h1>Log Analysis Report</h1><p>No logs available</p>")
            return json.dumps(summary)

        total_requests = len(df)
        success_rate = len(df[df["status"] == "success"]) / total_requests
        avg_response_time = df["response_time"].mean()
        report_md = f"""
# Log Analysis Report
- **Total Requests**: {total_requests}
- **Success Rate**: {success_rate:.2%}
- **Average Response Time**: {avg_response_time:.3f} seconds
"""
        report_html = markdown.markdown(report_md)
        
        # Load JSON reports
        reports = {}
        for report_file in ["build_report.json", "test_report.json", "deploy_report.json"]:
            try:
                if os.path.exists(report_file):
                    with open(report_file, "r") as f:
                        reports[report_file] = json.load(f)
                else:
                    reports[report_file] = {"status": "missing", "issues": [f"{report_file} not found"], "mitigations": ["Check previous job outputs"]}
            except Exception as e:
                reports[report_file] = {"status": "error", "issues": [f"Failed to load {report_file}: {str(e)}"], "mitigations": ["Verify file format"]}

        # Generate index.html
        index_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CI/CD Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        async function loadReport(file, elementId) {{
            try {{
                const response = await fetch(file);
                if (!response.ok) throw new Error('File not found');
                const data = await response.json();
                const table = document.getElementById(elementId);
                table.innerHTML = `
                    <tr><th class="border px-4 py-2">Status</th><td class="border px-4 py-2">${{data.status}}</td></tr>
                    <tr><th class="border px-4 py-2">Image</th><td class="border px-4 py-2">${{data.image || 'N/A'}}</td></tr>
                    <tr><th class="border px-4 py-2">Issues</th><td class="border px-4 py-2">${{data.issues?.join(', ') || 'None'}}</td></tr>
                    <tr><th class="border px-4 py-2">Mitigations</th><td class="border px-4 py-2">${{data.mitigations?.join(', ') || 'None'}}</td></tr>
                `;
            }} catch (error) {{
                document.getElementById(elementId).innerHTML = `<tr><td colspan="2">Error loading ${{file}}: ${{error.message}}</td></tr>`;
            }}
        }}
        window.onload = () => {{
            loadReport('build_report.json', 'build-report');
            loadReport('test_report.json', 'test-report');
            loadReport('deploy_report.json', 'deploy-report');
        }};
    </script>
</head>
<body class="bg-gray-100">
    <nav class="bg-blue-600 p-4 text-white">
        <ul class="flex space-x-4 justify-center">
            <li><a href="#build" class="hover:underline">Build</a></li>
            <li><a href="#test" class="hover:underline">Test</a></li>
            <li><a href="#deploy" class="hover:underline">Deploy</a></li>
            <li><a href="#log-analysis" class="hover:underline">Log Analysis</a></li>
        </ul>
    </nav>
    <div class="container mx-auto p-4">
        <h1 class="text-3xl font-bold text-center mb-4">CI/CD Dashboard</h1>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div id="build" class="bg-white p-4 rounded shadow">
                <h2 class="text-xl font-semibold mb-2">Build Report</h2>
                <table class="w-full border-collapse border"><tbody id="build-report"></tbody></table>
            </div>
            <div id="test" class="bg-white p-4 rounded shadow">
                <h2 class="text-xl font-semibold mb-2">Test Report</h2>
                <table class="w-full border-collapse border"><tbody id="test-report"></tbody></table>
            </div>
            <div id="deploy" class="bg-white p-4 rounded shadow">
                <h2 class="text-xl font-semibold mb-2">Deploy Report</h2>
                <table class="w-full border-collapse border"><tbody id="deploy-report"></tbody></table>
            </div>
            <div id="log-analysis" class="bg-white p-4 rounded shadow">
                <h2 class="text-xl font-semibold mb-2">Log Analysis Report</h2>
                <iframe src="log_analysis_report.html" class="w-full h-64 border"></iframe>
            </div>
        </div>
    </div>
</body>
</html>
"""
        try:
            with open("index.html", "w") as f:
                f.write(index_html)
        except Exception as e:
            print(f"Failed to write index.html: {str(e)}")
        try:
            with open("log_analysis_report.html", "w") as f:
                f.write(f"<html><body>{report_html}</body></html>")
        except Exception as e:
            print(f"Failed to write log_analysis_report.html: {str(e)}")
        
        summary = {
            "status": "success",
            "total_requests": total_requests,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "reports": reports
        }
        return json.dumps(summary)
    except Exception as e:
        print(f"Log analysis error: {str(e)}")
        summary = {"status": "failed", "issues": [f"Log analysis failed: {str(e)}"], "mitigations": ["Check logs and dependencies"]}
        try:
            with open("log_analysis_report.html", "w") as f:
                f.write(f"<h1>Log Analysis Report</h1><p>Error: {str(e)}</p>")
        except Exception as e:
            print(f"Failed to write log_analysis_report.html: {str(e)}")
        try:
            with open("index.html", "w") as f:
                f.write(index_html)
        except Exception as e:
            print(f"Failed to write index.html: {str(e)}")
        return json.dumps(summary)

# Register functions
try:
    print("Registering analyze_logs function...")
    log_analyst.register_for_execution()(analyze_logs)
    print("Function registered successfully")
except Exception as e:
    print(f"Failed to register function: {str(e)}")
    with open("log_analysis_report.html", "w") as f:
        f.write(f"<h1>Log Analysis Report</h1><p>Error: {str(e)}</p>")
    exit(1)

if __name__ == "__main__":
    try:
        print("Initiating chat to analyze logs...")
        autogen.initiate_chats([{
            "sender": autogen.UserProxyAgent(name="UserProxy"),
            "recipient": log_analyst,
            "message": "Analyze the logs and generate a report.",
            "max_turns": 1
        }])
        print("Chat initiated successfully")
    except Exception as e:
        print(f"Chat initiation failed: {str(e)}")
        summary = {"status": "failed", "issues": [f"Chat initiation failed: {str(e)}"], "mitigations": ["Check autogen and dependencies"]}
        try:
            with open("log_analysis_report.html", "w") as f:
                f.write(f"<h1>Log Analysis Report</h1><p>Error: {str(e)}</p>")
        except Exception as e:
            print(f"Failed to write log_analysis_report.html: {str(e)}")
        try:
            with open("index.html", "w") as f:
                f.write(index_html)
        except Exception as e:
            print(f"Failed to write index.html: {str(e)}")
        exit(1)
