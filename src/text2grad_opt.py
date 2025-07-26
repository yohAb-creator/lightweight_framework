import os
import re
import random
import time
import sys
from typing import Any, List
from absl import app, flags, logging
from android_world import registry
from android_world.env import env_launcher, json_action, interface
from android_world.agents import infer, base_agent

# --- Suppress verbose informational logs ---
logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

# --- Flag Definitions for standalone execution ---
flags.DEFINE_string('task_name_tg', 'ContactsAddContact,SystemWifiTurnOff,SystemBrightnessMax',
                    'Comma-separated list of android_world tasks to use for evaluation.')
flags.DEFINE_integer('num_mutations_tg', 5, 'Number of prompt variations to generate and test.')
flags.DEFINE_string('gcp_api_key_tg', os.environ.get("GCP_API_KEY"), 'Your Google Cloud API Key.')
flags.DEFINE_string('adb_path_tg', "C:\\Users\\yohan\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe",
                    'Path to ADB.')
flags.DEFINE_integer('console_port_tg', 5554, 'Emulator console port.')
flags.DEFINE_integer('grpc_port_tg', 8554, 'Emulator gRPC port.')
flags.DEFINE_integer('max_steps_tg', 25, 'Maximum steps for the agent per episode.')


def get_llm_response_tg(prompt: str) -> str:
    """Generic wrapper to call the LLM for Text2Grad."""
    if not FLAGS.gcp_api_key_tg:
        # Fallback to main key if available
        api_key = os.environ.get("GCP_API_KEY")
        if not api_key:
            return "ERROR: API Key not set"
        llm = infer.GeminiGcpWrapper('gemini-2.5-pro')
    else:
        llm = infer.GeminiGcpWrapper('gemini-2.5-pro')

    response, _, _ = llm.predict(prompt)
    return response


def parse_action_from_llm_response_tg(response_text: str, ui_elements: list[Any]) -> dict[str, Any] | None:
    """Extracts an action dictionary from the LLM's full response for Text2Grad."""
    action_str = response_text
    if "Action:" in response_text:
        action_str = response_text.split("Action:")[-1].strip()

    click_match = re.search(r'CLICK\("(.+?)"\)', action_str)
    if click_match:
        target_text = click_match.group(1)
        for i, element in enumerate(ui_elements):
            label = (element.content_description or element.text or "").strip()
            if target_text.lower() == label.lower():
                return {"action_type": "click", "index": i}
        return None

    type_match = re.search(r'TYPE\("(.+?)", "(.+?)"\)', action_str)
    if type_match:
        text_to_type = type_match.group(1)
        target_field_text = type_match.group(2)
        for i, element in enumerate(ui_elements):
            label = (element.content_description or element.text or "").strip()
            if target_field_text.lower() == label.lower() and element.is_editable:
                return {"action_type": "input_text", "text": text_to_type, "index": i}
        return None

    if "STATUS_COMPLETE" in action_str.upper():
        return {"action_type": "status", "goal_status": "complete"}

    return None


def mutate_prompt(base_prompt: str) -> str:
    """Creates a simple variation of a given prompt by prepending an instruction."""
    variations = [
        "Think step-by-step. ",
        "Be precise and logical. ",
        "What is the most direct next action to achieve the goal? ",
        "Analyze the screen carefully before acting. "
    ]
    prefix = random.choice(variations)
    return prefix + base_prompt


class TGAgent(base_agent.EnvironmentInteractingAgent):
    """A minimal agent for the Text2Grad reward function."""

    def __init__(self, env: interface.AsyncEnv, prompt_policy: str):
        super().__init__(env)
        self.prompt_policy = prompt_policy

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        state = self.get_post_transition_state()
        ui_elements = state.ui_elements
        clickable_elements = [f"- \"{(e.content_description or e.text or '').strip()}\"" for e in ui_elements if
                              e.is_clickable and (e.content_description or e.text)]
        observation_text = "\n".join(clickable_elements) if clickable_elements else "The screen is empty."

        current_prompt = self.prompt_policy.format(goal=goal, observation=observation_text,
                                                   history="")  # History not used in this simple agent
        llm_response = get_llm_response_tg(current_prompt)
        action_params = parse_action_from_llm_response_tg(llm_response, ui_elements)

        is_done = False
        if action_params:
            if action_params.get("action_type") == "status":
                is_done = True
            else:
                try:
                    action = json_action.JSONAction(**action_params)
                    self.env.execute_action(action)
                except Exception:
                    # Action failure is handled by the reward function
                    pass
        return base_agent.AgentInteractionResult(done=is_done, data={})


def reward_function(prompt_template: str, task_class) -> float:
    """Runs an agent and returns 1.0 for success, 0.0 for failure."""
    env = None
    try:
        env = env_launcher.load_and_setup_env(
            adb_path=FLAGS.adb_path_tg, console_port=FLAGS.console_port_tg, grpc_port=FLAGS.grpc_port_tg
        )
        task_instance = task_class(params=task_class.generate_random_params())
        goal = task_instance.goal
        task_instance.initialize_task(env)

        agent = TGAgent(env, prompt_template)
        agent.reset(go_home=True)
        time.sleep(2)

        for _ in range(FLAGS.max_steps_tg):
            result = agent.step(goal)
            if result.done:
                break
            time.sleep(1)

        return task_instance.is_successful(env)
    except Exception as e:
        print(f"An error occurred during reward calculation: {e}")
        return 0.0
    finally:
        if env:
            env.close()


def run_text2grad_optimization(training_tasks: List[Any] = None):
    """
    This function is imported and called by arpo_runner.py.
    It runs the Text2Grad optimization loop across a set of training tasks
    and returns the prompt with the best average performance.
    """
    print("\n" + "=" * 20 + " STAGE 2: Mixed-Task Text2Grad Optimization " + "=" * 20)

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

    if not training_tasks:
        print(" -> Running in standalone mode. Using tasks from flags.")
        task_names = FLAGS.task_name_tg.split(',')
        training_tasks = [aw_registry[name] for name in task_names if name in aw_registry]

    if not training_tasks:
        print("FATAL: No valid training tasks found for Text2Grad. Exiting.")
        # Return a default prompt if no tasks are available
        return """Your high-level goal is: {goal}
Here is a history of your previous actions:
{history}
Here are the available UI elements on the current screen:
{observation}
Based on your history and the current screen, what is your next action?
Respond ONLY in the format: Reason: [reason] Action: [action]"""

    base_template = """Your high-level goal is: {goal}
Here is a history of your previous actions:
{history}
Here are the available UI elements on the current screen:
{observation}
Based on your history and the current screen, what is your next action? If the task is complete, respond with STATUS_COMPLETE.
Respond ONLY in the format:
Reason: [Your reasoning here]
Action: [Your action here in the format CLICK("text"), OPEN_APP("App Name"), SCROLL("direction"), TYPE("text_to_type", "target_field_text"), or STATUS_COMPLETE]"""

    best_prompt = base_template
    best_avg_reward = 0.0

    # First, evaluate the baseline prompt
    print("\nEvaluating baseline prompt P0 across all training tasks...")
    baseline_rewards = [reward_function(base_template, task) for task in training_tasks]
    best_avg_reward = sum(baseline_rewards) / len(baseline_rewards) if baseline_rewards else 0.0
    print(f"\nBaseline Average Reward for P0: {best_avg_reward:.2%}")

    for i in range(FLAGS.num_mutations_tg):
        print(f"\n--- Mutation {i + 1}/{FLAGS.num_mutations_tg} ---")
        mutated_prompt = mutate_prompt(best_prompt)  # Mutate the current best

        print(f"  -> Evaluating mutated prompt across {len(training_tasks)} tasks...")
        rewards = [reward_function(mutated_prompt, task) for task in training_tasks]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"  -> Average Reward for Mutated Prompt: {avg_reward:.2%}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_prompt = mutated_prompt
            print("  -> Found a better prompt!")

    print("\n" + "=" * 25 + " Text2Grad Finished " + "=" * 25)
    print(f"Optimized Average Reward (P*): {best_avg_reward:.2%}")
    print("Returning best prompt found to ARPO runner.")
    return best_prompt


def main_tg(argv):
    """Main function for running this script standalone."""
    if FLAGS.gcp_api_key_tg:
        os.environ["GCP_API_KEY"] = FLAGS.gcp_api_key_tg

    best_prompt = run_text2grad_optimization()
    print("\n--- Final Optimized Prompt from Standalone Run ---")
    print(best_prompt)


if __name__ == "__main__" and "arpo_runner" not in sys.modules:
    app.parse_flags_with_usage(sys.argv)
    main_tg(sys.argv)
