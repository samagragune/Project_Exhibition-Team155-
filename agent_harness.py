# agent_harness.py
# Cognitive Reasoning Node: Manages frontier Vision-Language models (Qwen2.5-VL, DeepSeek-VL2, Gemini 2.0, LM Studio).
# Streams reasoning, validates 3D spatial tool calls, and handles multi-turn conversation memory.

import os
import sys
import base64
import json
from openai import OpenAI
import config


class AgentHarness:
    """Cognitive reasoning engine for robotic arm autonomy.
    Provides frontier visual grounding, 3D spatial planning, and safe tool execution."""

    def __init__(self, cfg: dict = config.agent_harness_CONFIG, tools: list = config.ROBOT_TOOLS):
        self.config = cfg
        self.tools = tools
        self.bounds = cfg.get("BOUNDS", {"X_MIN": 0.0, "X_MAX": 50.0, "Y_MIN": 0.0, "Y_MAX": 50.0, "Z_MIN": 0.0, "Z_MAX": 28.0})

        # Resolve active model configuration
        self.preset_name = self.config.get("ACTIVE_PRESET", "lm-studio")
        presets = self.config.get("PRESETS", {})
        active_preset = presets.get(self.preset_name, {})

        self.base_url = active_preset.get("base_url", self.config.get("LM_STUDIO_URL", "http://127.0.0.1:1234/v1"))
        self.api_key = active_preset.get("api_key", self.config.get("API_KEY", "lm-studio")) or "lm-studio"
        self.model_name = active_preset.get("model_name", self.config.get("MODEL_NAME", "gemma-4-12b-it@q4_k_m"))
        self.temperature = self.config.get("TEMPERATURE", 0.45)

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.history = [{"role": "system", "content": self.config.get("SYSTEM_PROMPT", config.sysPrompt)}]

        print(f"[AgentHarness] Initialized with preset '{self.preset_name}' (Model: {self.model_name} @ {self.base_url})")

    def switch_model_preset(self, preset_name: str):
        """Switches model provider (e.g. 'qwen2.5-vl', 'deepseek-vl2', 'gemini-2.0', 'lm-studio')."""
        presets = self.config.get("PRESETS", {})
        if preset_name in presets:
            self.preset_name = preset_name
            p = presets[preset_name]
            self.base_url = p.get("base_url", self.base_url)
            self.api_key = p.get("api_key", "lm-studio") or "lm-studio"
            self.model_name = p.get("model_name", self.model_name)
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            print(f"[AgentHarness] Switched model to '{preset_name}' ({self.model_name})")
            return True
        return False

    def _prune_past_images(self):
        """Removes older images to keep token context concise."""
        for msg in self.history:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                msg["content"] = [c for c in msg["content"] if c.get("type") != "image_url"]

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encodes local image to base64 JPEG string."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def validate_and_execute_tool(self, tool_name: str, args: dict) -> dict:
        """Validates tool arguments against 3D workspace safety bounds."""
        if tool_name == "move_arm":
            x, y, z = args.get("x_cm"), args.get("y_cm"), args.get("z_cm")
            if x is None or y is None or z is None:
                return {"status": "error", "message": "Missing x_cm, y_cm, or z_cm parameters."}

            try:
                x, y, z = float(x), float(y), float(z)
            except ValueError:
                return {"status": "error", "message": "Coordinates must be numerical floats."}

            if not (self.bounds["X_MIN"] <= x <= self.bounds["X_MAX"]):
                return {"status": "error", "message": f"X={x}cm violates bounds [{self.bounds['X_MIN']}, {self.bounds['X_MAX']}]"}
            if not (self.bounds["Y_MIN"] <= y <= self.bounds["Y_MAX"]):
                return {"status": "error", "message": f"Y={y}cm violates bounds [{self.bounds['Y_MIN']}, {self.bounds['Y_MAX']}]"}
            if not (self.bounds["Z_MIN"] <= z <= self.bounds["Z_MAX"]):
                return {"status": "error", "message": f"Z={z}cm violates height ceiling [{self.bounds['Z_MIN']}, {self.bounds['Z_MAX']}]"}

            return {"status": "legal", "message": f"Tool call verified within safe envelope: X={x}cm, Y={y}cm, Z={z}cm."}

        elif tool_name == "set_claw":
            state = str(args.get("state", "")).strip().lower()
            if state not in ["open", "closed"]:
                return {"status": "error", "message": f"Invalid claw state '{state}'. Must be 'open' or 'closed'."}
            return {"status": "pass", "message": f"Claw command '{state}' approved."}

        elif tool_name == "park_arm":
            pose = str(args.get("pose", "survey")).strip().lower()
            if pose not in ["survey", "rest"]:
                return {"status": "error", "message": f"Invalid pose '{pose}'. Must be 'survey' or 'rest'."}
            return {"status": "pass", "message": f"Park pose '{pose}' approved to clear camera."}

        return {"status": "error", "message": f"Unknown tool '{tool_name}'"}

    def _fallback_cognitive_decision(self, user_command: str, detected_objects: list) -> dict:
        """Autonomous fallback reasoning if external LLM endpoint is offline or booting."""
        cmd_lower = user_command.lower()
        print("\n[AgentHarness: Local Cognitive Engine] Resolving autonomous action...")

        # If user asks to park or clear view
        if "park" in cmd_lower or "clear" in cmd_lower or "survey" in cmd_lower or "out of the way" in cmd_lower:
            reasoning = "User requested clearing the camera frustum. Parking the robotic arm into 'survey' position outside the workspace viewing angle."
            tool_call = {"tool": "park_arm", "arguments": {"pose": "survey"}}
            feedback = self.validate_and_execute_tool(tool_call["tool"], tool_call["arguments"])
            return {
                "reasoning": reasoning,
                "response_text": "Moving arm to survey position to provide an unobstructed top-down view.",
                "tool_call": tool_call,
                "execution_feedback": feedback
            }

        # If user asks to open claw
        if "open" in cmd_lower and "claw" in cmd_lower:
            return {
                "reasoning": "Opening gripper claw as requested.",
                "response_text": "Releasing gripper claw.",
                "tool_call": {"tool": "set_claw", "arguments": {"state": "open"}},
                "execution_feedback": {"status": "pass", "message": "Claw open command approved."}
            }

        # If user asks to grab/pick an object
        if detected_objects and any(w in cmd_lower for w in ["grab", "pick", "take", "collect", "move", "sort", "approach"]):
            # Find matching object or default to first valid object
            target_obj = detected_objects[0]
            for obj in detected_objects:
                obj_id = str(obj.get("id"))
                if f"#{obj_id}" in cmd_lower or f"id:{obj_id}" in cmd_lower or f"object {obj_id}" in cmd_lower:
                    target_obj = obj
                    break

            tx = target_obj["centroid_cm"]["x"]
            ty = target_obj["centroid_cm"]["y"]
            tz = target_obj.get("grasp_recommended_z_cm", 2.5)

            # If prompt mentions approach from above
            if "approach" in cmd_lower or "above" in cmd_lower or "position" in cmd_lower:
                approach_z = min(22.0, target_obj.get("surface_top_z_cm", 4.0) + 12.0)
                reasoning = f"Initiating 2-stage grasp. Stage 1: Moving end-effector directly above target object #{target_obj['id']} to pre-grasp clearance height Z={approach_z}cm."
                tool_call = {"tool": "move_arm", "arguments": {"x_cm": tx, "y_cm": ty, "z_cm": approach_z}}
            else:
                reasoning = f"Targeting object #{target_obj['id']} at physical 3D location (X={tx}, Y={ty}, Z={tz}) using secondary depth sensor measurement."
                tool_call = {"tool": "move_arm", "arguments": {"x_cm": tx, "y_cm": ty, "z_cm": tz}}

            feedback = self.validate_and_execute_tool(tool_call["tool"], tool_call["arguments"])
            return {
                "reasoning": reasoning,
                "response_text": f"Aligning end-effector to 3D target coordinates of Object #{target_obj['id']}.",
                "tool_call": tool_call,
                "execution_feedback": feedback
            }

        # Default text response
        return {
            "reasoning": "Command analyzed. Workspace objects surveyed.",
            "response_text": f"Surveyed workspace: {len(detected_objects)} 3D objects mapped. Ready for your next instruction.",
            "tool_call": {"tool": "text_response", "arguments": {}},
            "execution_feedback": {"status": "info", "message": "Awaiting next task."}
        }

    def decide_action(self, user_command: str, image_path: str, detected_objects: list) -> dict:
        """Main cognitive cycle: sends 3D perception metadata and images to the LLM agent,
        streams tokens, parses tool calls, and returns structured execution payload."""
        base64_img = self._encode_image_to_base64(image_path)
        self._prune_past_images()

        current_user_content = []

        # Previous frame visual history for before/after comparison
        prev_image_path = os.path.join(os.path.dirname(image_path), "previous_annotated_output.jpg")
        if os.path.exists(prev_image_path):
            try:
                prev_base64 = self._encode_image_to_base64(prev_image_path)
                current_user_content.append({
                    "type": "text",
                    "text": "📷 [PREVIOUS FRAME] - State of workspace before your last action:"
                })
                current_user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{prev_base64}"}
                })
            except Exception:
                pass

        # 3D Depth Perception Context
        current_user_content.append({
            "type": "text",
            "text": (
                f"User Instruction: {user_command}\n\n"
                f"📷 [CURRENT FRAME] - 3D Object Perception & Secondary Depth Sensor Readings:\n"
                f"{json.dumps(detected_objects, indent=2)}\n\n"
                f"Analyze the spatial coordinates and issue a single atomic tool call: "
                f"`move_arm(x_cm, y_cm, z_cm)`, `set_claw(state='open'|'closed')`, or `park_arm(pose='survey'|'rest')`."
            )
        })
        current_user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })

        self.history.append({"role": "user", "content": current_user_content})

        print(f"\n[AgentHarness] Querying {self.model_name} @ {self.base_url}...")
        reasoning_text = ""
        content_text = ""
        tool_call_chunks = {}
        in_reasoning = False

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                tools=self.tools,
                tool_choice="auto",
                temperature=self.temperature,
                stream=True,
                timeout=12.0
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Reasoning tokens
                reasoning_chunk = getattr(delta, "reasoning_content", None)
                if reasoning_chunk:
                    in_reasoning = True
                    reasoning_text += reasoning_chunk
                    sys.stdout.write(f"\033[90m{reasoning_chunk}\033[0m")
                    sys.stdout.flush()

                # Content tokens
                if delta.content:
                    if in_reasoning:
                        sys.stdout.write("\n\n")
                        in_reasoning = False
                    content_text += delta.content
                    sys.stdout.write(delta.content)
                    sys.stdout.flush()

                # Tool call chunks
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_chunks:
                            tool_call_chunks[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function else "",
                                "arguments": ""
                            }
                        if tc.function and tc.function.name:
                            tool_call_chunks[idx]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_call_chunks[idx]["arguments"] += tc.function.arguments

            # Parse primary tool call
            parsed_tool = None
            parsed_args = {}
            execution_feedback = {"status": "info", "message": "No tool call requested."}

            if tool_call_chunks:
                first_tool = tool_call_chunks[0]
                call_id = first_tool["id"] or "call_1"
                parsed_tool = first_tool["name"]
                try:
                    parsed_args = json.loads(first_tool["arguments"])
                except Exception:
                    parsed_args = {}

                execution_feedback = self.validate_and_execute_tool(parsed_tool, parsed_args)

                self.history.append({
                    "role": "assistant",
                    "content": content_text or None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": parsed_tool,
                            "arguments": first_tool["arguments"]
                        }
                    }]
                })
                self.history.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(execution_feedback)
                })
            else:
                self.history.append({"role": "assistant", "content": content_text})

            return {
                "reasoning": reasoning_text,
                "response_text": content_text,
                "tool_call": {
                    "tool": parsed_tool or "text_response",
                    "arguments": parsed_args
                },
                "execution_feedback": execution_feedback
            }

        except Exception as err:
            print(f"\n[AgentHarness] Remote endpoint unavailable ({err}). Engaging local cognitive solver...")
            return self._fallback_cognitive_decision(user_command, detected_objects)


if __name__ == "__main__":
    harness = AgentHarness()
    mock_objs = [{
        "id": 1,
        "centroid_cm": {"x": 20.0, "y": 20.0, "z": 2.5},
        "surface_top_z_cm": 5.0,
        "height_cm": 5.0,
        "grasp_recommended_z_cm": 2.8
    }]
    test_img = config.agent_harness_CONFIG.get("TEST_IMAGE_PATH", "handshake/raw_capture.jpg")
    if os.path.exists(test_img):
        res = harness.decide_action("park the arm out of the camera view", test_img, mock_objs)
        print(f"\nResult: {res}")