import os
import re
import json
from typing import Any
from absl import app, flags
from android_world import registry
from android_world.agents import infer, t3a
from android_world.env import env_launcher
import google.generativeai as genai

FLAGS = flags.FLAGS

# Flag Definitions
flags.DEFINE_string(
    'adb_path',
    os.environ.get('adb_path', "C:\\Users\\yohan\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe"),
    'Path to the Android Debug Bridge (ADB).'
)
flags.DEFINE_integer('console_port', 5554, 'Console port of the running emulator.')
flags.DEFINE_integer('grpc_port', 8554, 'gRPC port of the running emulator.')
flags.DEFINE_string('gcp_api_key', os.environ.get("GCP_API_KEY"), 'Your Google Cloud API Key.')
flags.DEFINE_string('task_name', 'ContactsAddContact', 'Task to run')
flags.DEFINE_integer('max_steps', 15, 'Maximum number of agent steps per episode')
flags.DEFINE_enum('prompt_variant', 'zero_shot', ['zero_shot', 'few_shot', 'self_reflection'],
                  'The prompt variant to use for the LLM.')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)


def load_few_shot_examples():
    """Loads few-shot examples from a file using a robust path."""
    path = os.path.join(PROJECT_ROOT, "prompts", "few_shot_examples.txt")
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {path} not found. Running in zero-shot mode.")
        return ""


def load_prompt_template():
    """Loads the base prompt template from a file using a robust path."""
    path = os.path.join(PROJECT_ROOT, "prompts", "prompt_template.txt")
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {path} not found. Using fallback template.")
        # Fallback in case the file is missing
        fall_back_prompt = """Your goal is to: {goal}

Current screen:
- App: {app_name}
- UI Elements:
{ui_elements}

{instruction}"""
        return fall_back_prompt


def generate_prompt(goal: str, app_name: str, ui_elements: list[str], variant: str) -> str:
    """Configurable prompt template that supports different variants by loading a base template."""
    formatted_elements = "\n".join([f"- {elem.strip('\"')}" for elem in ui_elements])
    template = load_prompt_template()

    instruction = ""
    if variant == 'self_reflection':
        instruction = """First, explain your reasoning. Then, provide the single best next action on a new line.
Your response must be in the format:
Reason: [Your reasoning here]
Action: [Your action here in the format CLICK("element_text") or TYPE("text_to_type")]"""
    else:  # for zero_shot and few_shot
        instruction = """What is the single best next action? Respond in the format: CLICK("element_text") or TYPE("text_to_type")"""

    base_prompt = template.format(
        goal=goal,
        app_name=app_name,
        ui_elements=formatted_elements,
        instruction=instruction
    )

    if variant == 'few_shot':
        few_shot_examples = load_few_shot_examples()
        # Prepend examples to the main prompt
        return f"{few_shot_examples}\n{base_prompt}"

    return base_prompt


def prompt_llm(prompt: str, api_key: str) -> str:
    """Call Gemini API and return the model's response."""
    if not api_key:
        return "ERROR: GCP_API_KEY is not set."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    return response.text.strip()


def extract_clickable_ui(ui_elements: list[Any]) -> list[str]:
    """Extract list of unique clickable text descriptions from a list of UI elements."""
    clickable_texts = []
    if not ui_elements:
        return []
    for e in ui_elements:
        label = e.content_description or e.text
        if e.is_clickable and label:
            clean = label.replace("\n", " ").strip()
            if clean:
                clickable_texts.append(f'"{clean}"')
    return list(dict.fromkeys(clickable_texts))


def run_episode(task_class=None):
    """Runs one Android task episode."""
    env = None
    episode_log = []

    try:
        # --- FIX: Ensure API key is set as an environment variable ---
        # This makes the key available to all parts of the library, including the t3a agent.
        if FLAGS.gcp_api_key:
            os.environ["GCP_API_KEY"] = FLAGS.gcp_api_key
        elif not os.environ.get("GCP_API_KEY"):
            raise ValueError("GCP_API_KEY must be set either as an environment variable or via the --gcp_api_key flag.")

        print("🔧 Loading environment...")
        env = env_launcher.load_and_setup_env(
            adb_path=FLAGS.adb_path,
            console_port=FLAGS.console_port,
            grpc_port=FLAGS.grpc_port,
        )
        print("✅ Environment ready.")

        if task_class is None:
            task_registry = registry.TaskRegistry()
            aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
            task_class = aw_registry[FLAGS.task_name]

        task_instance = task_class(params=task_class.generate_random_params())
        goal = task_instance.goal

        print(f"\n📌 Task goal: {goal}")
        task_instance.initialize_task(env)
        print("📱 Task initialized.")

        # --- CORRECTED WRAPPER ---
        agent = t3a.T3A(env, infer.GeminiGcpWrapper('gemini-1.5-pro-latest'))

        for step_num in range(1, FLAGS.max_steps + 1):
            agent_response = agent.step(goal)
            latest_history = agent.history[-1]
            ui_elements_from_history = latest_history.get('before_element_list', [])
            raw_agent_output = latest_history.get('action_output', '')

            agent_action_text = "N/A"
            if '\nAction: ' in raw_agent_output:
                json_action_str = raw_agent_output.split('\nAction: ')[-1]
                try:
                    action_json = json.loads(json_action_str)
                    action_type = action_json.get("action_type")
                    if action_type == "click":
                        index = action_json.get("index")
                        if index is not None and index < len(ui_elements_from_history):
                            element = ui_elements_from_history[index]
                            label = (element.content_description or element.text or "").strip()
                            agent_action_text = f'CLICK("{label}")'
                    elif action_type == "input_text":
                        text = action_json.get("text", "")
                        agent_action_text = f'TYPE("{text}")'
                    else:
                        agent_action_text = json_action_str
                except (json.JSONDecodeError, IndexError):
                    agent_action_text = "Error parsing agent action"

            unique_elements = extract_clickable_ui(ui_elements_from_history)
            app_name_match = re.match(r'^[A-Z][a-z]*', task_class.__name__)
            app_name = app_name_match.group(0) if app_name_match else "Unknown"

            prompt_for_llm = generate_prompt(goal, app_name, unique_elements, FLAGS.prompt_variant)
            llm_full_response = prompt_llm(prompt_for_llm, FLAGS.gcp_api_key)

            # Parse reason and action for self-reflection
            llm_reason = "N/A"
            llm_action_text = llm_full_response
            if FLAGS.prompt_variant == 'self_reflection' and "Action:" in llm_full_response:
                parts = llm_full_response.split("Action:")
                llm_reason = parts[0].replace("Reason:", "").strip()
                llm_action_text = parts[1].strip()

            episode_log.append({
                "step": step_num,
                "prompt": prompt_for_llm,
                "llm_action": llm_action_text,
                "llm_reason": llm_reason,
                "agent_action": agent_action_text,
            })

            if agent_response.done:
                print(f"\n✅ Agent completed task at step {step_num}")
                break
        else:
            print(f"\n❌ Max steps reached without completing task.")

        success = task_instance.is_successful(env) == 1

        return {
            "goal": goal,
            "episode_log": episode_log,
            "success": success,
        }

    except Exception as e:
        print(f"\n🚨 ERROR: {e}")
        return {"goal": None, "episode_log": [], "success": False}
    finally:
        if env:
            print("\n🧹 Cleaning up environment...")
            env.close()


def main(argv):
    del argv
    run_episode()


if __name__ == "__main__":
    app.run(main)