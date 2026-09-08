# orchestrator.py
# Main Orchestrator: Rich TUI Autonomy Loop coordinating:
#   Camera Acquisition → Secondary Depth Fusion → SAM 2.1 3D Grounding → Frontier LLM Reasoning → Hardware Dispatch

import os
import sys
import json
import time
import shutil
import urllib.request
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

import config
from get_image import capture_and_save_frame, capture_depth_heatmap
from sam2_processor import SAM2Processor
from agent_harness import AgentHarness

console = Console()


class RoboticOrchestrator:
    """Coordinates perception-action cycles with 3D depth perception,
    frontier vision models, and multi-scenario environments."""

    def __init__(self, offline_mode: bool = False):
        self.offline_mode = offline_mode
        self.console = console
        self.sam_processor = SAM2Processor(config=config.sam2_processor_CONFIG)
        self.agent_harness = AgentHarness(cfg=config.agent_harness_CONFIG, tools=config.ROBOT_TOOLS)

    def print_banner(self):
        """Renders the cyber-aerospace system banner."""
        self.console.clear()
        banner_text = Text()
        banner_text.append("   ____  ____  ______  _  _______  ____  _______\n", style="bold cyan")
        banner_text.append("  / __ \\/ __ \\/ __/ / / / / ___/ / __ \\/ __/ _ \\\n", style="bold cyan")
        banner_text.append(" / /_/ / /_/ / _// /_/ / / /__/ / /_/ / _// ___/\n", style="bold blue")
        banner_text.append(" \\____/ .___/___/\\____/  \\___/  \\____/___/_/    \n", style="bold blue")
        banner_text.append("     /_/   ROBOTIC ARM 3D COGNITIVE AUTONOMY LAYER v3.0\n", style="dim white")

        self.console.print(Panel(banner_text, border_style="bold cyan", expand=False))

        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_row(
            f"[bold magenta]Model Preset:[/bold magenta] {self.agent_harness.preset_name} ({self.agent_harness.model_name})",
            f"[bold yellow]Depth Sensor:[/bold yellow] ACTIVE (3D Metric)",
            f"[bold green]Mode:[/bold green] {'OFFLINE (Simulated/Cache)' if self.offline_mode else 'ONLINE (PyBullet 3D Engine)'}"
        )
        self.console.print(info_table)
        self.console.print("[dim]─" * 80 + "[/dim]\n")

    def run_pipeline_step(self, user_command: str):
        """Executes one perception-action cycle:
        1. Capture RGB frame & Secondary Depth Sensor Heatmap.
        2. Perform 3D Perception & Depth Fusion (SAM 2.1 + metric elevation).
        3. Cognitive Reasoning & Tool Selection via Frontier LLM Agent.
        4. Hardware Tool Execution (move_arm, set_claw, park_arm)."""

        raw_image_path = "handshake/raw_capture.jpg"

        # Step 1: Vision & Depth Acquisition
        with self.console.status("[bold green]Step 1/3: Acquiring Visuals & Depth Sensor Stream...", spinner="dots"):
            if self.offline_mode:
                raw_image_path = config.agent_harness_CONFIG.get("TEST_IMAGE_PATH", "test_artifacts/test_image6.jpg")
                time.sleep(0.3)
            else:
                try:
                    raw_image_path = capture_and_save_frame(output_filename="raw_capture.jpg")
                    capture_depth_heatmap(output_filename="depth_heatmap.jpg")
                except Exception as e:
                    self.console.print(f"[bold red]❌ Camera Fetch Failed ({e}). Using cached frame...[/bold red]")
                    raw_image_path = config.agent_harness_CONFIG.get("TEST_IMAGE_PATH", "handshake/raw_capture.jpg")

        self.console.print(f"  [bold green]✓[/bold green] Perception Feed Acquired: [dim]{raw_image_path}[/dim]")

        # Step 2: 3D Depth Perception & Fusion
        output_image_path = config.sam2_processor_CONFIG.get("OUTPUT_IMAGE", "handshake/annotated_output.jpg")
        if os.path.exists(output_image_path):
            prev_image_path = os.path.join(os.path.dirname(output_image_path), "previous_annotated_output.jpg")
            try:
                shutil.copy(output_image_path, prev_image_path)
            except Exception:
                pass

        with self.console.status("[bold yellow]Step 2/3: Fusing 3D Depth Map with SAM 2.1 Segmentation...", spinner="dots"):
            # If running in same process or via simulator, depth map is auto-computed
            depth_map = None
            try:
                from simulator import sim_world
                depth_map = sim_world.get_metric_depth_map()
            except Exception:
                pass

            _, detected_objects = self.sam_processor.process_image(
                image_input=raw_image_path,
                depth_buffer=depth_map
            )

        # Render 3D Spatial Metadata HUD
        sam_table = Table(title="3D Multi-Sensor Perception Grid", show_header=True, header_style="bold magenta")
        sam_table.add_column("ID", style="cyan", width=5)
        sam_table.add_column("3D Centroid (X, Y, Z) cm", style="green")
        sam_table.add_column("Top Z", style="yellow", width=8)
        sam_table.add_column("Height", style="dim white", width=8)
        sam_table.add_column("Recommended Grasp Z", style="bold cyan")

        for obj in detected_objects:
            c = obj["centroid_cm"]
            sam_table.add_row(
                str(obj["id"]),
                f"({c['x']}, {c['y']}, {c.get('z', 2.0)})",
                f"{obj.get('surface_top_z_cm', 4.0)} cm",
                f"{obj.get('height_cm', 4.0)} cm",
                f"{obj.get('grasp_recommended_z_cm', 2.5)} cm"
            )

        self.console.print(Panel(sam_table, border_style="yellow", title="[bold yellow]Perception & Depth Fusion Layer[/bold yellow]"))

        # Step 3: Agent Reasoning
        self.console.print(Panel(f"[bold white]User Instruction:[/bold white] {user_command}", border_style="blue"))
        self.console.print("\n[bold cyan]🧠 Cognitive Reasoning Stream:[/bold cyan]")

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

                req = urllib.request.Request(sim_url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    feedback = resp_data
            except Exception as e:
                # Direct in-process fallback
                try:
                    from simulator import sim_world
                    if executed_tool == "move_arm":
                        args = tool_data.get("arguments", {})
                        feedback = sim_world.execute_move_arm(args["x_cm"], args["y_cm"], args["z_cm"])
                    elif executed_tool == "set_claw":
                        args = tool_data.get("arguments", {})
                        feedback = sim_world.execute_set_claw(args["state"])
                    elif executed_tool == "park_arm":
                        args = tool_data.get("arguments", {})
                        feedback = sim_world.execute_park_arm(args.get("pose", "survey"))
                except Exception as ex:
                    feedback = {"status": "error", "message": f"Dispatch failed: {e} / {ex}"}

        status_color = "green" if feedback.get("status") == "success" else "red" if feedback.get("status") == "error" else "yellow"

        tool_summary = Text()
        tool_summary.append("Tool Executed : ", style="bold white")
        tool_summary.append(f"{executed_tool}\n", style="bold cyan")
        tool_summary.append("Arguments     : ", style="bold white")
        tool_summary.append(f"{json.dumps(tool_data.get('arguments'))}\n", style="yellow")
        tool_summary.append("Hardware Echo : ", style="bold white")
        tool_summary.append(f"[{feedback.get('status', 'info').upper()}] {feedback.get('message', '')}", style=f"bold {status_color}")

        self.console.print(Panel(tool_summary, title="[bold green]🦾 Kinematic Execution Output[/bold green]", border_style=status_color))
        return executed_tool is not None and executed_tool != "text_response"

    def switch_scenario(self, scenario_name: str):
        """Switches the simulated scenario environment."""
        try:
            from simulator import sim_world
            sim_world.spawn_scenario(scenario_name)
            self.console.print(f"[bold green]Switched scenario to '{scenario_name}'![/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]Failed to switch scenario: {e}[/bold red]")

    def start_chat_loop(self):
        """Interactive autonomy loop with turn guardrails and slash commands."""
        self.print_banner()

        while True:
            try:
                user_input = self.console.input("\n[bold cyan]RoboAutonomy > [/bold cyan]").strip()
                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q", "/exit"]:
                    self.console.print("[bold red]Shutting down Autonomy Layer... Goodbye![/bold red]")
                    break

                if user_input.startswith("/scenario "):
                    sc_name = user_input.split(" ", 1)[1].strip()
                    self.switch_scenario(sc_name)
                    continue

                if user_input.startswith("/model "):
                    preset = user_input.split(" ", 1)[1].strip()
                    if self.agent_harness.switch_model_preset(preset):
                        self.console.print(f"[bold green]Model preset changed to '{preset}'![/bold green]")
                    else:
                        self.console.print(f"[bold red]Unknown preset '{preset}'. Available: {list(config.FRONTIER_PRESETS.keys())}[/bold red]")
                    continue

                if user_input.lower() in ["/clear", "clear"]:
                    self.print_banner()
                    continue

                step_command = user_input
                max_auto_turns = 10
                turn_count = 0

                while turn_count < max_auto_turns:
                    tool_was_executed = self.run_pipeline_step(user_command=step_command)
                    if not tool_was_executed:
                        break

                    turn_count += 1
                    self.console.print(f"\n[bold yellow]🔄 Visual Feedback Loop: Triggering Autonomous Turn {turn_count + 1}...[/bold yellow]\n")
                    step_command = "Inspect the updated camera view and depth sensor readings. Execute the next atomic action to progress or finish the task."

                if turn_count >= max_auto_turns:
                    self.console.print("[bold red]⚠️ Safety Limit: Autonomous turn ceiling reached.[/bold red]")

            except KeyboardInterrupt:
                self.console.print("\n[bold red]Interrupted by user. Exiting...[/bold red]")
                sys.exit(0)


if __name__ == "__main__":
    is_offline = "--offline" in sys.argv
    orchestrator = RoboticOrchestrator(offline_mode=is_offline)
    orchestrator.start_chat_loop()