import os
import json
from absl import app, flags
from .agent import run_episode

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 10, 'Number of episodes to run for evaluation.')

RESULTS_DIR = "results"


def save_results(results_data, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved detailed results to {path}")


def evaluate_episodes(num_episodes):
    all_results = []
    total_steps = 0
    total_correct_steps = 0
    total_successes = 0

    for ep_num in range(1, num_episodes + 1):
        print(f"\n{'=' * 20} 📱 Episode {ep_num} {'=' * 20}\n")
        result = run_episode()

        episode_log = result.get("episode_log", [])
        correct_steps_in_episode = 0
        for step in episode_log:
            llm_action = step.get("llm_action", "").strip()
            agent_action = step.get("agent_action", "").strip()

            if llm_action == agent_action:
                correct_steps_in_episode += 1
                step['exact_match'] = True
            else:
                step['exact_match'] = False

        result['step_accuracy'] = correct_steps_in_episode / len(episode_log) if episode_log else 0
        all_results.append(result)

        total_steps += len(episode_log)
        total_correct_steps += correct_steps_in_episode
        if result.get("success", False):
            total_successes += 1

        print(f"\n--- Episode {ep_num} Summary ---")
        print(f"Goal: {result.get('goal', 'N/A')}")
        print(f"Success: {result.get('success', False)}")
        print(f"Step Accuracy: {result['step_accuracy']:.2%}")

    overall_success_rate = total_successes / num_episodes if num_episodes > 0 else 0
    overall_step_accuracy = total_correct_steps / total_steps if total_steps > 0 else 0

    print("\n" + "=" * 25 + " 📊 FINAL EVALUATION SUMMARY " + "=" * 25)
    print(f"Prompt Variant Used: {FLAGS.prompt_variant}")
    print(f"Total Episodes Run: {num_episodes}")
    print(f"Overall Episode Success Rate: {total_successes}/{num_episodes} = {overall_success_rate:.2%}")
    print(f"Overall Step Accuracy: {total_correct_steps}/{total_steps} = {overall_step_accuracy:.2%}")
    print("=" * 80)

    # Create a summary dictionary to save in the JSON file
    summary_data = {
        "prompt_variant": FLAGS.prompt_variant,
        "num_episodes": num_episodes,
        "overall_success_rate": overall_success_rate,
        "overall_step_accuracy": overall_step_accuracy,
        "overall_success_rate_formatted": f"{overall_success_rate:.2%}",
        "overall_step_accuracy_formatted": f"{overall_step_accuracy:.2%}",
        "total_successes": total_successes,
        "total_steps": total_steps,
        "total_correct_steps": total_correct_steps
    }

    # Combine summary and per-episode results into a single dictionary
    final_results = {
        "summary": summary_data,
        "episodes": all_results
    }

    # Save results with a filename indicating the prompt variant used
    results_filename = f"evaluation_results_{FLAGS.prompt_variant}.json"
    save_results(final_results, results_filename)


def main(argv):
    del argv
    evaluate_episodes(FLAGS.num_episodes)


if __name__ == "__main__":
    app.run(main)