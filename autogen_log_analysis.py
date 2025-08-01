import autogen
from autogen import ConversableAgent
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import sqlite3
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import markdown
import subprocess

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

# Define agents
log_analyst = autogen.AssistantAgent(
    name="LogAnalyst",
    llm_config=False,
    system_message="Analyze KinD logs, summarize anomalies (e.g., HTTP errors, login failures), and suggest mitigations. Return JSON summary."
)
log_analyst.llm_client = CustomLLMClient()

report_generator = autogen.AssistantAgent(
    name="ReportGenerator",
    llm_config=False,
    system_message="Convert Log Analyst's JSON summary into a markdown report with sections: Log Summary, Issues, Mitigations."
)
report_generator.llm_client = CustomLLMClient()

# Generate mock logs
def generate_mock_logs(num_logs=50):
    timestamps = [datetime.now() - timedelta(minutes=x) for x in range(num_logs)][::-1]
    log_levels = ['INFO'] * int(0.7 * num_logs) + ['WARNING'] * int(0.2 * num_logs) + ['ERROR'] * int(0.1 * num_logs)
    status_codes = [200] * int(0.8 * num_logs) + [404] * int(0.15 * num_logs) + [500] * int(0.05 * num_logs)
    response_times = [max(10, min(1000, int(x))) for x in np.random.normal(100, 50, num_logs)]
    min_length = min(len(timestamps), len(log_levels), len(status_codes), len(response_times))
    timestamps = timestamps[:min_length]
    log_levels = log_levels[:min_length]
    status_codes = status_codes[:min_length]
    response_times = response_times[:min_length]
    messages = []
    for i in range(min_length):
        level = log_levels[i]
        code = status_codes[i]
        pod_name = f"microservice-pod-{i%3}"
        if level == 'ERROR':
            messages.append(f"[{pod_name}] HTTP {code} error: {'Internal Server Error' if code == 500 else 'Not Found'}")
        elif level == 'WARNING':
            messages.append(f"[{pod_name}] High response time detected: {response_times[i]:.2f}ms")
        else:
            messages.append(f"[{pod_name}] Request processed successfully")
    return pd.DataFrame({
        'Timestamp': timestamps,
        'LogLevel': log_levels,
        'StatusCode': status_codes,
        'ResponseTime': response_times,
        'Message': messages
    })

# Store training summary
def store_training_summary(agent_name, summary):
    try:
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS training_data
                     (agent_name TEXT, timestamp TEXT, summary TEXT)''')
        c.execute('INSERT INTO training_data VALUES (?, ?, ?)',
                  (agent_name, datetime.now().isoformat(), json.dumps(summary)))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

# Run log analysis
def run_log_analysis():
    # Try to fetch real logs
    try:
        log_result = subprocess.run(
            ["kubectl", "logs", "-l", "app=microservice", "-n", "default"],
            capture_output=True, text=True, check=True
        )
        log_text = log_result.stdout
    except subprocess.CalledProcessError:
        print("Falling back to mock logs due to log fetch failure")
        logs_df = generate_mock_logs()
        log_text = "\n".join([f"{row.Timestamp} {row.LogLevel} {row.Message}" for _, row in logs_df.iterrows()])

    # Analyze logs
    analysis_prompt = f"Analyze the following logs and return a JSON summary of issues and mitigations:\n{log_text[:1000]}"
    analysis_response = log_analyst.llm_client.create({"prompt": analysis_prompt})
    try:
        summary = json.loads(analysis_response["choices"][0]["message"]["content"])
    except json.JSONDecodeError:
        summary = {"issues": ["Failed to parse analysis"], "mitigations": ["Check GPT-2 output"]}

    # Generate markdown report
    markdown_prompt = f"""Convert the following JSON summary into a markdown report with sections: Log Summary, Issues, Mitigations.
```json
{json.dumps(summary, indent=2)}
```"""
    markdown_response = report_generator.llm_client.create({"prompt": markdown_prompt})
    markdown_content = json.loads(markdown_response["choices"][0]["message"]["content"]).get("message", "")
    
    # Convert markdown to HTML
    try:
        html_body = markdown.markdown(markdown_content) if markdown_content.strip() else "<h1>Error</h1><p>No valid markdown content generated.</p>"
    except Exception as e:
        print(f"Markdown conversion error: {e}")
        html_body = "<h1>Error</h1><p>Failed to generate report.</p>"

    # Generate log_analysis_report.html
    log_html = f"""
    <html>
    <head>
        <title>Log Analysis Report</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
            .container {{ max-width: 800px; margin: auto; }}
        </style>
    </head>
    <body>
        <div class="container">{html_body}</div>
    </body>
    </html>
    """
    with open("log_analysis_report.html", "w", encoding="utf-8") as f:
        f.write(log_html)

    # Generate index.html
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CI/CD Pipeline Reports</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { background-color: #f9fafb; }
            .report-card { transition: transform 0.3s; }
            .report-card:hover { transform: scale(1.02); }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #e5e7eb; padding: 8px; text-align: left; }
            th { background-color: #f3f4f6; }
        </style>
    </head>
    <body class="font-sans">
        <!-- Navigation Bar -->
        <nav class="bg-indigo-600 text-white p-4 sticky top-0 shadow-md">
            <div class="container mx-auto flex justify-between items-center">
                <h1 class="text-xl font-bold">Pipeline Reports</h1>
                <div class="space-x-4">
                    <a href="#build" class="hover:underline">Build</a>
                    <a href="#test" class="hover:underline">Test</a>
                    <a href="#deploy" class="hover:underline">Deploy</a>
                    <a href="#log" class="hover:underline">Log Analysis</a>
                </div>
            </div>
        </nav>

        <!-- Hero Section -->
        <section class="bg-indigo-50 py-12 text-center">
            <div class="container mx-auto">
                <h2 class="text-3xl font-bold text-gray-800">CI/CD Pipeline Report Dashboard</h2>
                <p class="mt-4 text-gray-600">View detailed reports from your build, test, deployment, and log analysis processes.</p>
            </div>
        </section>

        <!-- Reports Section -->
        <section class="container mx-auto py-12 grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Build Report -->
            <div id="build" class="report-card bg-white p-6 rounded-lg shadow-lg">
                <h3 class="text-xl font-semibold text-gray-800 mb-4">Build Report</h3>
                <div id="build-report-content">
                    <p class="text-gray-500">Loading...</p>
                </div>
            </div>

            <!-- Test Report -->
            <div id="test" class="report-card bg-white p-6 rounded-lg shadow-lg">
                <h3 class="text-xl font-semibold text-gray-800 mb-4">Test Report</h3>
                <div id="test-report-content">
                    <p class="text-gray-500">Loading...</p>
                </div>
            </div>

            <!-- Deploy Report -->
            <div id="deploy" class="report-card bg-white p-6 rounded-lg shadow-lg">
                <h3 class="text-xl font-semibold text-gray-800 mb-4">Deploy Report</h3>
                <div id="deploy-report-content">
                    <p class="text-gray-500">Loading...</p>
                </div>
            </div>

            <!-- Log Analysis Report -->
            <div id="log" class="report-card bg-white p-6 rounded-lg shadow-lg">
                <h3 class="text-xl font-semibold text-gray-800 mb-4">Log Analysis Report</h3>
                <div id="log-report-content">
                    <iframe src="log_analysis_report.html" class="w-full h-96 border-none"></iframe>
                </div>
            </div>
        </section>

        <!-- Footer -->
        <footer class="bg-gray-800 text-white text-center p-4 mt-12">
            <p>&copy; 2025 CI/CD Pipeline. Powered by GitHub Pages.</p>
        </footer>

        <!-- JavaScript to Load JSON Reports -->
        <script>
            // Function to render JSON report as a table
            function renderReport(data, containerId) {
                const container = document.getElementById(containerId);
                if (!data || Object.keys(data).length === 0) {
                    container.innerHTML = '<p class="text-red-500">No data available.</p>';
                    return;
                }

                let html = '<table>';
                for (const [key, value] of Object.entries(data)) {
                    html += `<tr><th>${key}</th><td>${Array.isArray(value) ? value.join(', ') : value}</td></tr>`;
                }
                html += '</table>';
                container.innerHTML = html;
            }

            // Fetch and display JSON reports
            async function loadReports() {
                try {
                    const buildReport = await fetch('build_report.json').then(res => res.ok ? res.json() : {});
                    renderReport(buildReport, 'build-report-content');

                    const testReport = await fetch('test_report.json').then(res => res.ok ? res.json() : {});
                    renderReport(testReport, 'test-report-content');

                    const deployReport = await fetch('deploy_report.json').then(res => res.ok ? res.json() : {});
                    renderReport(deployReport, 'deploy-report-content');
                } catch (error) {
                    console.error('Error loading reports:', error);
                    ['build-report-content', 'test-report-content', 'deploy-report-content'].forEach(id => {
                        document.getElementById(id).innerHTML = '<p class="text-red-500">Failed to load report.</p>';
                    });
                }
            }

            // Load reports on page load
            window.onload = loadReports;
        </script>
    </body>
    </html>
    """
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(index_html)

    # Validate file creation
    for file in ["log_analysis_report.html", "index.html"]:
        if os.path.exists(file) and os.path.getsize(file) > 0:
            print(f"✅ {file} created successfully.")
        else:
            print(f"❌ {file} was not created or is empty.")

    store_training_summary("log_analyst", summary)

if __name__ == "__main__":
    run_log_analysis()
