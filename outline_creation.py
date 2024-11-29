import os

# Step 1: Define the directory structure
project_structure = {
    "project-root": {
        "Phase1_Basic_Chatbot": {
            "actions": {},
            "data": {
                "nlu.yml": "",
                "domain.yml": "",
                "stories.yml": "",
                "rules.yml": ""
            },
            "models": {},
            "static": {
                "chatbot.css": "",
                "chatbot.js": "",
                "index.html": ""
            },
            "README.md": "# Phase 1: Basic chatbot implementation",
            "app.py": "",
            "config.yml": "",
            "endpoints.yml": "",
            "requirements.txt": "",
            "Dockerfile": ""
        },
        "Phase2_Advanced_Functionalities": {
            "actions": {
                "actions.py": "",
                "order_status_api.py": ""
            },
            "data": {
                "nlu.yml": "",
                "domain.yml": "",
                "stories.yml": "",
                "rules.yml": ""
            },
            "models": {},
            "static": {
                "chatbot.css": "",
                "chatbot.js": "",
                "chatbot-theme.css": "",
                "index.html": ""
            },
            "logs": {
                "action_logs.log": "",
                "user_interactions.log": ""
            },
            "README.md": "# Phase 2: Advanced functionalities",
            "app.py": "",
            "config.yml": "",
            "endpoints.yml": "",
            "requirements.txt": "",
            "Dockerfile": "",
            "docker-compose.yml": ""
        },
        "Phase3_Premium_Features": {
            "actions": {
                "actions.py": "",
                "sentiment_analysis.py": "",
                "proactive_notifications.py": ""
            },
            "data": {
                "nlu.yml": "",
                "domain.yml": "",
                "stories.yml": "",
                "rules.yml": ""
            },
            "models": {},
            "static": {
                "chatbot.css": "",
                "chatbot.js": "",
                "chatbot-theme.css": "",
                "index.html": ""
            },
            "app": {
                "app.py": "",
                "live_chat.py": "",
                "proactive_tasks.py": ""
            },
            "analytics": {
                "dashboard.py": "",
                "dashboard.html": "",
                "reports": {
                    "user_interaction_report.csv": "",
                    "sentiment_analysis_report.csv": ""
                }
            },
            "logs": {
                "sentiment_logs.log": "",
                "proactive_notifications.log": ""
            },
            "test_cases": {
                "test_nlu.py": "",
                "test_dialogue.py": ""
            },
            "README.md": "# Phase 3: Premium features",
            "config.yml": "",
            "endpoints.yml": "",
            "requirements.txt": "",
            "Dockerfile": "",
            "docker-compose.yml": ""
        },
        "LICENSE": "",
        "README.md": "# Main Project Documentation",
        ".gitignore": "",
        "CHANGELOG.md": "# Log of changes between phases"
    }
}

# Step 2: Function to create directories and files
def create_project_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):  # Create a directory
            os.makedirs(path, exist_ok=True)
            create_project_structure(path, content)
        else:  # Create a file
            with open(path, 'w') as file:
                file.write(content)

# Step 3: Execute the function to create the structure
base_dir = os.path.abspath("project-root")
create_project_structure(base_dir, project_structure)

print(f"Project structure created at {base_dir}")
