import autogen
import json
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
        f.write(f"""
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Analysis Report</title>
    <link rel="stylesheet" href="output.css">
</head>
<body>
    <h1 class="text-2xl font-bold text-center text-gray-800 mt-4">Log Analysis Report</h1>
    <p class="text-center text-red-500">Error: Failed to initialize GPT-2: {str(e)}</p>
</body>
</html>
""")
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
        system_message="You are a Log Analyst. Analyze logs and generate an HTML report."
    )
    log_analyst.llm_client = CustomLLMClient()
    print("LogAnalyst initialized successfully")
except Exception as e:
    print(f"Failed to initialize LogAnalyst: {str(e)}")
    with open("log_analysis_report.html", "w") as f:
        f.write(f"""
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Analysis Report</title>
    <link rel="stylesheet" href="output.css">
</head>
<body>
    <h1 class="text-2xl font-bold text-center text-gray-800 mt-4">Log Analysis Report</h1>
    <p class="text-center text-red-500">Error: Failed to initialize LogAnalyst: {str(e)}</p>
</body>
</html>
""")
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

# Index.html content
index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CI/CD Pipeline Reports</title>
    <link rel="stylesheet" href="output.css">
</head>
<body class="font-sans">
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
    <section class="bg-indigo-50 py-12 text-center">
        <div class="container mx-auto">
            <h2 class="text-3xl font-bold text-gray-800">CI/CD Pipeline Report Dashboard</h2>
            <p class="mt-4 text-gray-600">View detailed reports from your build, test, deployment, and log analysis processes.</p>
        </div>
    </section>
    <section class="container mx-auto py-12 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div id="build" class="report-card bg-white p-6 rounded-lg shadow-lg">
            <h3 class="text-xl font-semibold text-gray-800 mb-4">Build Report</h3>
            <div id="build-report-content">
                <p class="text-gray-500">Loading...</p>
            </div>
        </div>
        <div id="test" class="report-card bg-white p-6 rounded-lg shadow-lg">
            <h3 class="text-xl font-semibold text-gray-800 mb-4">Test Report</h3>
            <div id="test-report-content">
                <p class="text-gray-500">Loading...</p>
            </div>
        </div>
        <div id="deploy" class="report-card bg-white p-6 rounded-lg shadow-lg">
            <h3 class="text-xl font-semibold text-gray-800 mb-4">Deploy Report</h3>
            <div id="deploy-report-content">
                <p class="text-gray-500">Loading...</p>
            </div>
        </div>
        <div id="log" class="report-card bg-white p-6 rounded-lg shadow-lg">
            <h3 class="text-xl font-semibold text-gray-800 mb-4">Log Analysis Report</h3>
            <div id="log-report-content">
                <iframe src="log_analysis_report.html" class="w-full h-96 border-none" onerror="this.parentElement.innerHTML='<p class=\\'text-red-500\\'>Failed to load log analysis report: File not found</p>'"></iframe>
            </div>
        </div>
    </section>
    <footer class="bg-gray-800 text-white text-center p-4 mt-12">
        <p>&copy; 2025 CI/CD Pipeline. Powered by GitHub Pages.</p>
    </footer>
    <script>
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
        async function loadReports() {
            try {
                console.log('Fetching build_report.json...');
                const buildResponse = await fetch('build_report.json', { cache: 'no-store' });
                if (!buildResponse.ok) throw new Error(`HTTP ${buildResponse.status} for build_report.json`);
                const buildReport = await buildResponse.json();
                console.log('Loaded build_report.json:', buildReport);
                renderReport(buildReport, 'build-report-content');

                console.log('Fetching test_report.json...');
                const testResponse = await fetch('test_report.json', { cache: 'no-store' });
                if (!testResponse.ok) throw new Error(`HTTP ${testResponse.status} for test_report.json`);
                const testReport = await testResponse.json();
                console.log('Loaded test_report.json:', testReport);
                renderReport(testReport, 'test-report-content');

                console.log('Fetching deploy_report.json...');
                const deployResponse = await fetch('deploy_report.json', { cache: 'no-store' });
                if (!deployResponse.ok) throw new Error(`HTTP ${deployResponse.status} for deploy_report.json`);
                const deployReport = await deployResponse.json();
                console.log('Loaded deploy_report.json:', deployReport);
                renderReport(deployReport, 'deploy-report-content');
            } catch (error) {
                console.error('Error loading reports:', error.message);
                ['build-report-content', 'test-report-content', 'deploy-report-content'].forEach(id => {
                    document.getElementById(id).innerHTML = `<p class="text-red-500">Failed to load report: ${error.message}</p>`;
                });
            }
        }
        window.onload = loadReports;
    </script>
</body>
</html>
"""

# Analyze logs
def analyze_logs(_):
    summary = {"status": "unknown", "issues": [], "mitigations": []}
    try:
        print("Starting log analysis...")
        df = generate_mock_logs()
        if df.empty:
            summary["status"] = "failed"
            summary["issues"].append("No logs generated")
            summary["mitigations"].append("Check log generation logic")
            report_html = """
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Analysis Report</title>
    <link rel="stylesheet" href="output.css">
</head>
<body>
    <h1 class="text-2xl font-bold text-center text-gray-800 mt-4">Log Analysis Report</h1>
    <p class="text-center text-red-500">No logs available</p>
</body>
</html>
"""
        else:
            total_requests = len(df)
            success_rate = len(df[df["status"] == "success"]) / total_requests
            avg_response_time = df["response_time"].mean()
            report_html = f"""
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Analysis Report</title>
    <link rel="stylesheet" href="output.css">
</head>
<body>
    <h1 class="text-2xl font-bold text-center text-gray-800 mt-4">Log Analysis Report</h1>
    <table class="max-w-2xl mx-auto my-5">
        <tr><th>Total Requests</th><td>{total_requests}</td></tr>
        <tr><th>Success Rate</th><td>{success_rate:.2%}</td></tr>
        <tr><th>Average Response Time</th><td>{avg_response_time:.3f} seconds</td></tr>
    </table>
</body>
</html>
"""
            summary["status"] = "success"
            summary["total_requests"] = total_requests
            summary["success_rate"] = success_rate
            summary["avg_response_time"] = avg_response_time

        # Load JSON reports
        reports = {}
        for report_file in ["build_report.json", "test_report.json", "deploy_report.json"]:
            try:
                print(f"Loading {report_file}...")
                if os.path.exists(report_file):
                    with open(report_file, "r") as f:
                        reports[report_file] = json.load(f)
                    print(f"Loaded {report_file}: {json.dumps(reports[report_file], indent=2)}")
                else:
                    print(f"{report_file} not found")
                    reports[report_file] = {"status": "missing", "issues": [f"{report_file} not found"], "mitigations": ["Check previous job outputs"]}
            except Exception as e:
                print(f"Failed to load {report_file}: {str(e)}")
                reports[report_file] = {"status": "error", "issues": [f"Failed to load {report_file}: {str(e)}"], "mitigations": ["Verify file format"]}
            summary["reports"] = reports

        # Write index.html
        try:
            print(f"Writing index.html to {os.path.abspath('index.html')}")
            with open("index.html", "w") as f:
                f.write(index_html)
            print("Wrote index.html successfully")
        except Exception as e:
            print(f"Failed to write index.html: {str(e)}")
            summary["issues"].append(f"Failed to write index.html: {str(e)}")
            summary["mitigations"].append("Check disk space and permissions")

        # Write log_analysis_report.html
        try:
            print(f"Writing log_analysis_report.html to {os.path.abspath('log_analysis_report.html')}")
            with open("log_analysis_report.html", "w") as f:
                f.write(report_html)
            print("Wrote log_analysis_report.html successfully")
        except Exception as e:
            print(f"Failed to write log_analysis_report.html: {str(e)}")
            summary["issues"].append(f"Failed to write log_analysis_report.html: {str(e)}")
            summary["mitigations"].append("Check disk space and permissions")

        return(partition_summary)
    except Exception as e:
        print(f"Log analysis error: {str(e)}")
        summary["status"] = "failed"
        summary["issues"].append(f"Log analysis failed: {str(e)}")
        summary["mitigations"].append("Check logs and dependencies")
        report_html = f"""
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Analysis Report</title>
    <link rel="stylesheet" href="output.css">
</head>
<body>
    <h1 class="text-2xl font-bold text-center text-gray-800 mt-4">Log Analysis Report</h1>
    <p class="text-center text-red-500">Error: Log analysis failed: {str(e)}</p>
</body>
</html>
"""
        try:
            print(f"Writing log_analysis_report.html to {os.path.abspath('log_analysis_report.html')}")
            with open("log_analysis_report.html", "w") as f:
                f.write(report_html)
            print("Wrote log_analysis_report.html for error")
        except Exception as e:
            print(f"Failed to write log_analysis_report.html: {str(e)}")
        try:
            print(f"Writing index.html to {os.path.abspath('index.html')}")
            with open("index.html", "w") as f:
                f.write(index_html)
            print("Wrote index.html for error")
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
        f.write(f"""
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Analysis Report</title>
    <link rel="stylesheet" href="output.css">
</head>
<body>
    <h1 class="text-2xl font-bold text-center text-gray-800 mt-4">Log Analysis Report</h1>
    <p class="text-center text-red-500">Error: Failed to register function: {str(e)}</p>
</body>
</html>
""")
    exit(1)

if __name__ == "__main__":
    try:
        print("Initiating chat to analyze logs...")
        user_proxy = autogen.UserProxyAgent(name="UserProxy")
        result = analyze_logs(user_proxy)
        print(f"analyze_logs result: {result}")
        autogen.initiate_chats([{
            "sender": user_proxy,
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
                f.write(f"""
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Analysis Report</title>
    <link rel="stylesheet" href="output.css">
</head>
<body>
    <h1 class="text-2xl font-bold text-center text-gray-800 mt-4">Log Analysis Report</h1>
    <p class="text-center text-red-500">Error: Chat initiation failed: {str(e)}</p>
</body>
</html>
""")
            print("Wrote log_analysis_report.html for error")
        except Exception as e:
            print(f"Failed to write log_analysis_report.html: {str(e)}")
        try:
            with open("index.html", "w") as f:
                f.write(index_html)
            print("Wrote index.html for error")
        except Exception as e:
            print(f"Failed to write index.html: {str(e)}")
        exit(1)