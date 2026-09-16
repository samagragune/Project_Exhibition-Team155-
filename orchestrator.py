# orchestrator.py
# Helper script for the Robotic Arm Autonomy Layer.
# Provides the perception-action pipeline as a reusable Python module.
# The frontend calls this logic through FastAPI endpoints (/api/chat, /switch_scenario).
# Can also be run standalone as a simple CLI.

import os
import sys
import json
import time
import shutil
import config
from get_image import capture_and_save_frame, capture_depth_heatmap
from sam2_processor import SAM2Processor
from agent_harness import AgentHarness


class RoboticOrchestrator:
    """Coordinates perception-action cycles: capture → segment → reason → execute."""

    def __init__(self, offline_mode: bool = False):
        self.offline_mode = offline_mode
        self.sam_processor = SAM2Processor(config=config.sam2_processor_CONFIG)
        self.agent_harness = AgentHarness(cfg=config.agent_harness_CONFIG, tools=config.ROBOT_TOOLS)

    def run_pipeline_step(self, user_command: str):
        """Executes one perception-action cycle.
        Returns (executed_tool_name_or_None, detected_objects)."""
        raw_image_path = "handshake/raw_capture.jpg"

        # Step 1: Vision & Depth Acquisition
        if self.offline_mode:
            raw_image_path = config.agent_harness_CONFIG.get("TEST_IMAGE_PATH", "test_artifacts/test_image6.jpg")
            time.sleep(0.3)
        else:
            try:
                raw_image_path = capture_and_save_frame(output_filename="raw_capture.jpg")
                capture_depth_heatmap(output_filename="depth_heatmap.jpg")
            except Exception as e:
                print(f"[Orchestrator] Camera fetch failed ({e}). Using cached frame.")
                raw_image_path = config.agent_harness_CONFIG.get("TEST_IMAGE_PATH", "handshake/raw_capture.jpg")

        # Step 2: 3D Depth Perception & Fusion
        output_image_path = config.sam2_processor_CONFIG.get("OUTPUT_IMAGE", "handshake/annotated_output.jpg")
        if os.path.exists(output_image_path):
            prev_image_path = os.path.join(os.path.dirname(output_image_path), "previous_annotated_output.jpg")
            try:
                shutil.copy(output_image_path, prev_image_path)
            except Exception:
                pass

        depth_map = None
        try:
            from simulator import get_sim_world
            depth_map = get_sim_world().get_metric_depth_map()
        except Exception:
            pass

        _, detected_objects = self.sam_processor.process_image(
            image_input=raw_image_path,
            depth_buffer=depth_map
        )

        # Step 3: Agent Reasoning
        output_payload = self.agent_harness.decide_action(
            user_command=user_command,
            image_path=output_image_path if os.path.exists(output_image_path) else raw_image_path,
            detected_objects=detected_objects
        )

        # Step 4: Dispatch Tool Call to Simulator Backend
        tool_data = output_payload.get("tool_call", {})
        executed_tool = tool_data.get("tool")
        feedback = output_payload.get("execution_feedback", {})

        if executed_tool and executed_tool != "text_response" and not self.offline_mode:
            try:
                sim_url = f"http://{config.simulator_CONFIG['HOST']}:{config.simulator_CONFIG['PORT']}/execute_tool"
                payload = json.dumps({
                    "tool": executed_tool,
                    "arguments": tool_data.get("arguments", {})
                }).encode("utf-8")

                import urllib.request
                req = urllib.request.Request(sim_url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    feedback = resp_data
            except Exception as e:
                try:
                    from simulator import get_sim_world
                    sw = get_sim_world()
                    if executed_tool == "move_arm":
                        args = tool_data.get("arguments", {})
                        feedback = sw.execute_move_arm(args["x_cm"], args["y_cm"], args["z_cm"])
                    elif executed_tool == "set_claw":
                        args = tool_data.get("arguments", {})
                        feedback = sw.execute_set_claw(args["state"])
                    elif executed_tool == "park_arm":
                        args = tool_data.get("arguments", {})
                        feedback = sw.execute_park_arm(args.get("pose", "survey"))
                except Exception as ex:
                    feedback = {"status": "error", "message": f"Dispatch failed: {e} / {ex}"}

        output_payload["execution_feedback"] = feedback
        return executed_tool, output_payload, detected_objects

    def switch_scenario(self, scenario_name: str):
        """Switches the simulated scenario environment."""
        try:
            from simulator import get_sim_world
            get_sim_world().spawn_scenario(scenario_name)
            print(f"[Orchestrator] Switched scenario to '{scenario_name}'.")
        except Exception as e:
            print(f"[Orchestrator] Failed to switch scenario: {e}")
            raise


def run_autonomy_loop(command: str, max_turns: int = None):
    """Standalone autonomy loop. Used by __main__ entry point."""
    if max_turns is None:
        max_turns = config.simulator_CONFIG.get("MAX_AUTO_TURNS", 30)

    orchestrator = RoboticOrchestrator()
    turn_count = 0
    step_command = command

    while turn_count < max_turns:
        executed_tool, output_payload, detected_objects = orchestrator.run_pipeline_step(
            user_command=step_command
        )

        tool_name = output_payload.get("tool_call", {}).get("tool", "text_response")
        feedback = output_payload.get("execution_feedback", {})
        print(f"  Tool: {tool_name} | Feedback: {feedback.get('status', 'unknown')} - {feedback.get('message', '')}")

        if tool_name == "text_response":
            break

        turn_count += 1
        print(f"  Auto-turn {turn_count}/{max_turns}: Triggering next action...")
        step_command = "Inspect the updated camera view and depth sensor readings. Execute the next atomic action to progress or finish the task."

    if turn_count >= max_turns:
        print(f"[Orchestrator] Safety limit: {max_turns} autonomous turns reached.")


if __name__ == "__main__":
    is_offline = "--offline" in sys.argv
    max_turns = config.simulator_CONFIG.get("MAX_AUTO_TURNS", 30)

    if len(sys.argv) > 1 and sys.argv[1].startswith("/scenario "):
        sc_name = sys.argv[1].split(" ", 1)[1].strip()
        orchestrator = RoboticOrchestrator(offline_mode=is_offline)
        orchestrator.switch_scenario(sc_name)
    elif len(sys.argv) > 1:
        run_autonomy_loop(command=" ".join(sys.argv[1:]), max_turns=max_turns)
    else:
        print(f"Usage: python orchestrator.py <command> [--offline]")
        print(f"       python orchestrator.py /scenario <scenario_name> [--offline]")
        print(f"Max auto turns: {max_turns}")
        sys.exit(1)
