"""
Enhanced multi‑agent pipeline using the CrewAI framework to orchestrate
OpenMed clinical language models.  This script demonstrates how you
might structure a collection of specialist agents that work together
to analyse free‑form clinical text, select the most appropriate
OpenMed models, run those models on the text and summarise the
findings.  It is designed for demonstration purposes – it builds the
CrewAI agents and tasks, but it also executes the pipeline directly
without invoking CrewAI, so you can experiment immediately.

Prerequisites
--------------
Install the ``openmed`` and ``crewai`` packages.  CrewAI is used here
only to illustrate how the agents and tasks could be defined.  The
``crewai`` package is *not* installed in this environment by default,
so this script will still run because it does not call ``crew.kickoff``.

Usage
-----
Run this script from the command line and pass a piece of clinical text
to analyse.  The pipeline will automatically select the appropriate
OpenMed models, execute them and summarise the results::

    python3 multi_agent_pipeline_crewai.py "Patient diagnosed with acute lymphoblastic leukemia and started on imatinib."

If you have installed CrewAI and configured a compatible LLM, you can
uncomment the call to ``crew.kickoff`` at the bottom of the script to
execute the tasks via CrewAI.

This script is provided for educational purposes and does not
constitute medical advice.  Do not use it to make clinical decisions.
"""

import argparse
from typing import Dict, List, Any, Iterable, Optional

try:
    # CrewAI is an optional dependency.  The imports are placed in a
    # try/except block so that the script can run even if crewai is not
    # installed.  When installed, these imports provide Agent, Task and
    # Crew classes for orchestrating multiple agents.
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    # If CrewAI is not available, we'll set a flag and continue.  The
    # pipeline itself does not depend on crewai, so the script will
    # still function.
    CREWAI_AVAILABLE = False

# Import OpenMed functions.  The ``openmed`` package provides
# ``get_model_suggestions`` to select appropriate models based on text,
# and ``analyze_text`` to run a specific model on text.
try:
    from openmed import get_model_suggestions, analyze_text
except ImportError as exc:
    raise ImportError(
        "This script requires the openmed package. Please install it via 'pip install openmed'."
    ) from exc


def detect_device() -> str:
    """Detect the best available compute device.

    Returns ``"cuda"`` if a CUDA device is available, ``"mps"`` if the
    Apple Metal Performance Shader is available (macOS), otherwise
    defaults to CPU (``"cpu"``).
    """
    # We import torch lazily because not all environments have it
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        # If torch is not installed or there's any other issue, fall back to CPU
        pass
    return "cpu"


class OpenMedPipeline:
    """Pipeline encapsulating model selection, execution and summarisation.

    This class exposes three primary methods:

    * ``route_models`` – given a piece of text, call the OpenMed API to
      retrieve recommended models.
    * ``run_models`` – run each selected model on the text and return a
      dictionary mapping model names to their extracted entities.
    * ``summarise`` – compute a human‑readable summary of the
      detection results, grouping entities by type and model.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or detect_device()

    def route_models(self, text: str) -> List[str]:
        """Choose appropriate OpenMed models for the given text.

        It uses ``openmed.get_model_suggestions`` to analyse the
        text and returns a list of model names suggested by OpenMed.

        Parameters
        ----------
        text: str
            The clinical text to analyse.

        Returns
        -------
        List[str]
            A list of model names.
        """
        response = get_model_suggestions(
            text=text,
            temperature=0.0,
        )
        # The API returns a list of suggestions; each suggestion has a
        # ``model_name`` field.  We extract unique names, preserving order.
        model_names: List[str] = []
        for suggestion in response.get("model_suggestions", []):
            name = suggestion.get("model_name")
            if name and name not in model_names:
                model_names.append(name)
        return model_names

    def run_models(self, text: str, model_names: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Run multiple OpenMed models on the provided text.

        Parameters
        ----------
        text: str
            The clinical text to analyse.
        model_names: Iterable[str]
            Names of OpenMed models to run.

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            A mapping from model names to the list of detected entities.
        """
        results: Dict[str, List[Dict[str, Any]]] = {}
        for model_name in model_names:
            try:
                # Run the model via OpenMed's ``analyze_text`` API.  We
                # specify the device to offload computation to GPU or
                # accelerate on Apple Silicon where available.  If the
                # model cannot be loaded on the selected device, the
                # library will fall back to CPU.
                entities = analyze_text(
                    model_name=model_name,
                    text=text,
                    device=self.device,
                )
                results[model_name] = entities
            except Exception as exc:
                # Log the error for debugging and continue with other models.
                print(f"Warning: failed to run model {model_name}: {exc}")
        return results

    def summarise(self, results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Create a plain‑text summary of extracted entities.

        The summary groups entities by their type (label) and lists
        unique entity texts for each model.  It produces a human‑readable
        string suitable for printing or logging.

        Parameters
        ----------
        results: dict
            Mapping from model names to lists of entity dicts.

        Returns
        -------
        str
            A formatted string summarising the extraction results.
        """
        lines: List[str] = []
        for model_name, entities in results.items():
            lines.append(f"Model {model_name} detected:")
            # Group entities by label
            groups: Dict[str, List[str]] = {}
            for ent in entities:
                label = ent.get("label", "Unknown")
                text = ent.get("text", "")
                groups.setdefault(label, []).append(text)
            # Format groups
            for label, texts in groups.items():
                # Remove duplicates while preserving order
                seen = set()
                unique_texts = []
                for t in texts:
                    if t not in seen:
                        unique_texts.append(t)
                        seen.add(t)
                joined_texts = ", ".join(unique_texts)
                lines.append(f"  • {label}: {joined_texts}")
            lines.append("")
        return "\n".join(lines)


def build_crewai_objects(pipeline: OpenMedPipeline):
    """Build CrewAI agents, tasks and crew for the pipeline.

    This helper function constructs three agents and three tasks:

    * The **Model Selector** chooses appropriate models for the text.
    * The **Analysis Runner** executes the selected models.
    * The **Summariser** produces a human‑friendly summary.

    The returned objects can be used with CrewAI's runtime to execute
    the tasks via an LLM, provided that a compatible LLM and
    configuration are available.  Note that the tasks rely on the
    pipeline's methods; if executed through CrewAI, the agents will
    generate natural language reasoning to decide how to call those
    methods.

    Parameters
    ----------
    pipeline: OpenMedPipeline
        The pipeline instance whose methods will be referenced in
        task descriptions.

    Returns
    -------
    dict
        A dictionary containing the created agents, tasks and crew.
    """
    if not CREWAI_AVAILABLE:
        raise RuntimeError(
            "CrewAI is not installed. Install it via 'pip install crewai' to use this function."
        )

    # Define the agents.  Each agent is given a role, a goal and a
    # backstory.  The ``allow_code_execution`` flag enables the agent to
    # execute Python code when required.  In production you might want
    # to restrict code execution or provide custom tools instead.
    selector_agent = Agent(
        role="Model Selector",
        goal=(
            "Identify the most relevant OpenMed models to apply to a given "
            "piece of clinical text. Use the provided pipeline methods to "
            "determine which models are best suited for the task."
        ),
        backstory=(
            "You are an expert in clinical NLP and have extensive knowledge "
            "of the OpenMed model suite. Your job is to choose the right "
            "models that will extract meaningful entities from the input text."
        ),
        allow_code_execution=True,
    )

    analysis_agent = Agent(
        role="Analysis Runner",
        goal=(
            "Execute the selected OpenMed models on the provided text and "
            "record the extracted entities. Use the pipeline's run_models "
            "method to perform the actual computation."
        ),
        backstory=(
            "You are a computational agent that knows how to run machine "
            "learning models efficiently. You take a list of model names and "
            "apply them to the input to extract entities."
        ),
        allow_code_execution=True,
    )

    summariser_agent = Agent(
        role="Summariser",
        goal=(
            "Summarise the results of the entity extraction in a clear and "
            "concise way. Group entities by their type and indicate which "
            "models detected them. Use the pipeline's summarise method to "
            "format the results."
        ),
        backstory=(
            "You are skilled at synthesising complex information into human "
            "readable summaries. After the models have been run, you present "
            "the findings in a way that clinicians can easily interpret."
        ),
        allow_code_execution=True,
    )

    # Define tasks.  Tasks reference the pipeline methods by describing
    # what needs to be done.  In a real CrewAI execution, the agents
    # would interpret these descriptions and call the appropriate
    # functions.  Here we specify the context so that each task depends
    # on the previous one's output.
    select_task = Task(
        description=(
            "Given the input clinical text, use the pipeline's ``route_models`` "
            "method to determine which OpenMed models should be run. Return a "
            "list of model names."
        ),
        expected_output=(
            "A list of OpenMed model names such as `disease_detection_tiny` or "
            "`pharma_detection_superclinical`."
        ),
        agent=selector_agent,
    )

    analysis_task = Task(
        description=(
            "Run all selected models on the clinical text using the "
            "pipeline's ``run_models`` method. You must take the output of "
            "the model selection task as input. Return a dictionary mapping "
            "model names to lists of detected entities."
        ),
        expected_output=(
            "A JSON‑serialisable dictionary where each key is a model name and "
            "each value is a list of entities returned by that model."
        ),
        agent=analysis_agent,
        context=[select_task],
    )

    summarise_task = Task(
        description=(
            "Generate a readable summary of the entity detection results. Use "
            "the pipeline's ``summarise`` method to format the output. You "
            "should group entities by their label (e.g., disease, drug) and "
            "indicate which model detected them."
        ),
        expected_output=(
            "A multi‑line string summarising the results. Each model should be "
            "listed with bullet points for each entity type and their values."
        ),
        agent=summariser_agent,
        context=[analysis_task],
        markdown=False,
    )

    crew = Crew(
        agents=[selector_agent, analysis_agent, summariser_agent],
        tasks=[select_task, analysis_task, summarise_task],
        process=Process.sequential,
        verbose=True,
    )

    return {
        "agents": {
            "selector": selector_agent,
            "analysis": analysis_agent,
            "summariser": summariser_agent,
        },
        "tasks": {
            "select_task": select_task,
            "analysis_task": analysis_task,
            "summarise_task": summarise_task,
        },
        "crew": crew,
    }


def main() -> None:
    # Parse command‑line arguments
    parser = argparse.ArgumentParser(
        description="Multi‑agent pipeline for running OpenMed models using CrewAI."
    )
    parser.add_argument(
        "text",
        help="Clinical text to analyse",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional compute device to use (cpu, cuda or mps). "
            "If omitted, the script will auto‑detect the best available device."
        ),
    )
    args = parser.parse_args()

    # Build the pipeline
    pipeline = OpenMedPipeline(device=args.device)

    # Route models based on the input text
    model_names = pipeline.route_models(args.text)
    if not model_names:
        print("No models were suggested for the given text. Exiting.")
        return

    print("Selected models:")
    for name in model_names:
        print(f"  • {name}")

    # Run the models and collect results
    results = pipeline.run_models(args.text, model_names)
    # Summarise the results
    summary = pipeline.summarise(results)
    print("\nSummary of extracted entities:")
    print(summary)

    # Optionally build CrewAI objects for demonstration purposes
    if CREWAI_AVAILABLE:
        # Construct agents, tasks and crew.  This is optional and
        # demonstrates how you might integrate the pipeline into a
        # multi‑agent architecture.  Uncomment the lines below to run
        # the crew via CrewAI (requires configured LLM and API key).
        crew_objects = build_crewai_objects(pipeline)
        crew = crew_objects["crew"]
        # Uncomment to execute via CrewAI.  This will use your default
        # configured LLM to step through each task.
        # result = crew.kickoff(inputs={"input_text": args.text})
        # print("\nCrewAI result:")
        # print(result)


if __name__ == "__main__":
    main()
