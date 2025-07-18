# LLM Agent Evaluation Framework for Android World

This project contains a lightweight framework for evaluating the performance of Large Language Model (LLM) agents on the `android_world` benchmark.

The framework is designed to:

* Run an expert agent (`t3a`) on a series of tasks in a live Android emulator.

* Simultaneously prompt a custom LLM with the same observations.

* Compare the actions of the custom LLM against the expert agent to calculate a "step accuracy" score.

* Test and compare different prompting strategies (Zero-Shot, Few-Shot, Self-Reflection).

* Log detailed results for analysis.

## 1. Setup

### Prerequisites

* Python 3.11+

* [Anaconda/Conda](https://www.anaconda.com/download)

* [Android Studio](https://developer.android.com/studio) with a configured Android Virtual Device (AVD).

* A Google Cloud Platform (GCP) API key with the "Generative Language API" enabled.

### Installation Steps

1. **Clone the `android_world` Repository:**
   This project relies on the `android_world` library. Clone it into a separate directory.

git clone https://github.com/google-research/android_world.git


2. **Create and Activate Conda Environment:**

conda create -n android_world python=3.11
conda activate android_world


3. **Install `android_world` Library:**
Navigate to the cloned `android_world` directory and install it in editable mode. This is required to apply a necessary workaround.

cd path/to/android_world
pip install -e .


4. **Install Project Dependencies:**
Navigate back to this project's root directory and install its requirements.

cd path/to/your/evaluation_framework
pip install -r requirements.txt

Note: You may need to create a requirements.txt file with 'absl-py', 'google-generativeai', etc.


5. **Set API Key:**
Set your GCP API key as an environment variable in your terminal session.

In PowerShell

$env:GCP_API_KEY = "your-gcp-api-key"


### **Required Workaround for Windows**

The `android_world` library has a known issue where its setup process fails on Windows. To resolve this, a one-line code modification is required in the installed library.

* **File to Edit:** `.../android_world/android_world/env/android_world_controller.py`

* **Modification:** In the `__init__` method of the `AndroidWorldController` class, comment out the line that calls `apply_a11y_forwarder_app_wrapper`.

self._env = apply_a11y_forwarder_app_wrapper(

self._env, adb_controller, grpc_port

)


## 2. Usage

### Step 1: Launch the Emulator

Before running the evaluation, you must start the Android emulator from a terminal with the correct gRPC port.

In PowerShell (replace YourName)

$emulator_path = "C:\Users\YourName\AppData\Local\Android\Sdk\emulator\emulator.exe"
$avd_name = "AndroidWorldAvd"
& $emulator_path -avd $avd_name -no-snapshot -grpc 8554


### Step 2: Run the Evaluation

Once the emulator is running, you can run the evaluation script from the project's root directory. Use the flags to control the number of episodes and the prompting strategy.

**Run a 10-episode evaluation using the `few_shot` prompt variant:**

python -m src.evaluation --num_episodes=10 --prompt_variant=few_shot


**Run all three prompt variants sequentially:**

python -m src.evaluation --num_episodes=10 --prompt_variant=zero_shot; python -m src.evaluation --num_episodes=10 --prompt_variant=few_shot; python -m src.evaluation --num_episodes=10 --prompt_variant=self_reflection


### Step 3: View Results

The script will print a summary to the console and save detailed logs to the `results/` directory in a file named `evaluation_results_[variant_name].json`.
