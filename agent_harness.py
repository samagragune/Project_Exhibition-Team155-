# agent_harness.py
# LLM client module — manages conversation history, queries the model via
# OpenAI-compatible API, validates tool calls against workspace bounds,
# and streams reasoning tokens to stdout.

import os
import base64
import json
import sys
import config
from openai import OpenAI

class AgentHarness:
    """Manages the LLM conversation loop: builds prompts with images + metadata,
    streams model responses (reasoning + tool calls), validates arguments against
    workspace bounds, and maintains persistent conversation history."""

    def __init__(self, cfg: dict = config.agent_harness_CONFIG, tools: list = config.ROBOT_TOOLS):
        self.config = cfg
        self.tools = tools
        self.bounds = cfg.get("BOUNDS", {"X_MIN": 0.0, "X_MAX": 50.0, "Y_MIN": 0.0, "Y_MAX": 50.0, "Z_MIN": 0.0, "Z_MAX": 30.0})

        # Resolve active model configuration from presets
        self.preset_name = cfg.get("ACTIVE_PRESET", "lm-studio")
        presets = cfg.get("PRESETS", {})
        active_preset = presets.get(self.preset_name, {})

        self.base_url = active_preset.get("base_url", cfg.get("LM_STUDIO_URL", "http://127.0.0.1:1234/v1"))
        self.api_key = active_preset.get("api_key", cfg.get("API_KEY", "lm-studio"))
        self.model_name = active_preset.get("model_name", cfg.get("MODEL_NAME", "gemma-4-12b-it@q4_k_m"))
        self.temperature = cfg.get("TEMPERATURE", 0.55)

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=120.0
        )
        # Persistent history initialized with system prompt
        self.history = [{"role": "system", "content": self.config.get("SYSTEM_PROMPT", config.sysPrompt)}]

        print(f"[AgentHarness] Initialized with preset '{self.preset_name}' (Model: {self.model_name} @ {self.base_url})")

    def switch_model_preset(self, preset_name: str):
        """Switches model provider (e.g. 'qwen2.5-vl', 'deepseek-vl2', 'gemini-2.0', 'lm-studio')."""
        presets = self.config.get("PRESETS", {})
        if preset_name in presets:
            self.preset_name = preset_name
            p = presets[preset_name]
            self.base_url = p.get("base_url", self.base_url)
            self.api_key = p.get("api_key", self.config.get("API_KEY", "lm-studio"))
            self.model_name = p.get("model_name", self.model_name)
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=120.0
            )
            print(f"[AgentHarness] Switched model to '{preset_name}' ({self.model_name})")
            return True
        return False

    def _prune_past_images(self):
        """Keeps images from the current and previous user messages only.
        Strips base64 image data from all older user messages to reduce
        context window size while preserving before/after visual context."""
        user_messages_with_images = []
        for msg in self.history:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                has_images = any(c.get("type") == "image_url" for c in msg["content"])
                if has_images:
                    user_messages_with_images.append(msg)

        keep_ids = set()
        if len(user_messages_with_images) >= 1:
            keep_ids.add(id(user_messages_with_images[-1]))
        if len(user_messages_with_images) >= 2:
            keep_ids.add(id(user_messages_with_images[-2]))

        for msg in self.history:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                if id(msg) not in keep_ids:
                    msg["content"] = [c for c in msg["content"] if c.get("type") != "image_url"]

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Reads a local JPEG file and returns its Base64-encoded string
        representation for inclusion in the LLM prompt."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def validate_and_execute_tool(self, tool_name: str, args: dict) -> dict:
        """Validates tool call arguments against workspace bounds and tool schema.
        Returns a status dict ('legal', 'pass', or 'error') with a description message.
        Does NOT execute the tool — that is handled by the orchestrator/simulator."""
        if tool_name == "move_arm":
            x, y, z = args.get("x_cm"), args.get("y_cm"), args.get("z_cm")
            
            if x is None or y is None or z is None:
                return {"status": "error", "message": "Hardware Fault: Missing x_cm, y_cm, or z_cm parameters."}

            # Boundary Check
            if not (self.bounds["X_MIN"] <= x <= self.bounds["X_MAX"]):
                return {"status": "error", "message": f"Hardware Fault: X={x}cm violates bounds [{self.bounds['X_MIN']}, {self.bounds['X_MAX']}]"}
            if not (self.bounds["Y_MIN"] <= y <= self.bounds["Y_MAX"]):
                return {"status": "error", "message": f"Hardware Fault: Y={y}cm violates bounds [{self.bounds['Y_MIN']}, {self.bounds['Y_MAX']}]"}
            if not (self.bounds["Z_MIN"] <= z <= self.bounds["Z_MAX"]):
                return {"status": "error", "message": f"Hardware Fault: Z={z}cm violates bounds [{self.bounds['Z_MIN']}, {self.bounds['Z_MAX']}]"}
            
            return {"status": "legal", "message": f"tool call was within legal limits of motion (X:{x}cm, Y:{y}cm, Z:{z}cm)."}

        elif tool_name == "set_claw":
            state = args.get("state")
            if state not in ["open", "closed"]:
                return {"status": "error", "message": f"Hardware Fault: Invalid claw state '{state}'."}
            return {"status": "pass", "message": f"tool call was of correct format '{state}'."}

        elif tool_name == "park_arm":
            pose = args.get("pose")
            if pose not in ["survey", "rest"]:
                return {"status": "error", "message": f"Hardware Fault: Invalid park pose '{pose}'. Use 'survey' or 'rest'."}
            return {"status": "pass", "message": f"tool call was of correct format '{pose}'."}

        return {"status": "error", "message": f"Unknown tool: {tool_name}"}

    def decide_action(self, user_command: str, image_path: str, detected_objects: list) -> dict:
        """Queries the LLM with the current frame, previous frame (if available),
        SAM 2.1 metadata, and user command. Streams reasoning and tool call tokens
        to stdout. Validates the tool call against workspace bounds.
        Returns a dict with reasoning text, response text, tool call, and feedback."""
        base64_img = self._encode_image_to_base64(image_path)
        
        # 1. Strip past base64 images to keep context window light
        self._prune_past_images()

        # 2. Build current turn user payload (Previous frame + Current frame + Metadata)
        current_user_content = []

        # Check for previous frame artifact to provide before/after visual context
        prev_image_path = os.path.join(os.path.dirname(image_path), "previous_annotated_output.jpg")
        if os.path.exists(prev_image_path):
            prev_base64 = self._encode_image_to_base64(prev_image_path)
            current_user_content.append({
                "type": "text",
                "text": "📷 [PREVIOUS FRAME] - State of workspace before your last action:"
            })
            current_user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{prev_base64}"}
            })

        # Attach current frame and metadata
        current_user_content.append({
            "type": "text", 
            "text": f"User Command / Step: {user_command}\n\n📷 [CURRENT FRAME] - SAM 2.1 Metadata:\n{json.dumps(detected_objects, indent=2)}\n\nCompare both frames to evaluate what changed and decide the next single tool call."
        })
        current_user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })
        
        # 3. Append to persistent conversation history
        self.history.append({"role": "user", "content": current_user_content})

        print(f"\n[AgentHarness] Connected to LM Studio ({self.config['MODEL_NAME']})")
        print("="*60)
        print("🧠 STREAMING MODEL REASONING & OUTPUT:")
        print("-" * 60)

        # Storage for streamed outputs
        reasoning_text = ""
        content_text = ""
        tool_call_chunks = {}
        in_reasoning = False

        try:
            # Enable Streaming via OpenAI client - NOW PASSING self.history!
            stream = self.client.chat.completions.create(
                model=self.config["MODEL_NAME"],
                messages=self.history,
                tools=self.tools,
                tool_choice="auto",
                temperature=self.config["TEMPERATURE"],
                stream=True
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta

                # 1. Capture Reasoning / Thinking tokens
                reasoning_chunk = getattr(delta, "reasoning_content", None)
                if reasoning_chunk:
                    in_reasoning = True
                    reasoning_text += reasoning_chunk
                    sys.stdout.write(f"\033[90m{reasoning_chunk}\033[0m")  # Dim Gray output
                    sys.stdout.flush()

                # 2. Capture Content tokens
                if delta.content:
                    if in_reasoning:
                        sys.stdout.write("\n\n")
                        in_reasoning = False

                    content_text += delta.content
                    sys.stdout.write(delta.content)
                    sys.stdout.flush()

                # 3. Capture Streaming Tool Calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        index = tc.index
                        if index not in tool_call_chunks:
                            tool_call_chunks[index] = {
                                "id": tc.id or "",
                                "name": tc.function.name or "" if tc.function else "",
                                "arguments": ""
                            }
                        
                        if tc.function and tc.function.name:
                            tool_call_chunks[index]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_call_chunks[index]["arguments"] += tc.function.arguments

            print("\n" + "-" * 60)

            # Reconstruct the tool call (strictly processing only the FIRST tool call per turn)
            parsed_tool = None
            parsed_args = {}
            execution_feedback = {"status": "info", "message": "No tool call requested."}

            if tool_call_chunks:
                first_tool = tool_call_chunks[0]
                call_id = first_tool["id"] or "call_1"
                parsed_tool = first_tool["name"]
                try:
                    parsed_args = json.loads(first_tool["arguments"])
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Failed to parse tool arguments as JSON. "
                        f"Tool: '{parsed_tool}', Raw arguments: {first_tool['arguments']!r}"
                    )

                # Run hardware validation
                execution_feedback = self.validate_and_execute_tool(parsed_tool, parsed_args)

                # Record Assistant Tool Call in conversation history
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

                # Record mandatory Tool Execution Output in conversation history
                self.history.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(execution_feedback)
                })

                print("\n🛠️ EXECUTED TOOL CALL:")
                print(f"   Function : {parsed_tool}")
                print(f"   Arguments: {json.dumps(parsed_args)}")
                print(f"   Result   : [{execution_feedback['status'].upper()}] {execution_feedback['message']}")

            else:
                # Text-only response, append standard assistant message
                self.history.append({"role": "assistant", "content": content_text})

            print("="*60 + "\n")

            # Structured payload return for the future Orchestrator / Chat UI
            return {
                "reasoning": reasoning_text,
                "response_text": content_text,
                "tool_call": {
                    "tool": parsed_tool or "text_response",
                    "arguments": parsed_args
                },
                "execution_feedback": execution_feedback
            }

        except Exception as e:
            print(f"\n[AgentHarness] Error during streaming execution: {e}")
            raise e


# ==============================================================================
# 🚀 TEST STREAMING HARNESS WITH "punch the can"
# ==============================================================================
if __name__ == "__main__":
    harness = AgentHarness()
    
    # Mock SAM 2.1 Metadata for test_image6.jpg
    sample_sam_output = [
        {
            "id": 1,
            "center_cm": {"x": 15.2, "y": 10.5},
            "center_px": {"x": 320, "y": 240},
            "bbox_px": [220, 140, 420, 340]
        }
    ]

    test_image_file = config.agent_harness_CONFIG["TEST_IMAGE_PATH"]
    test_user_prompt = "punch the can"

    if os.path.exists(test_image_file):
        output = harness.decide_action(
            user_command=test_user_prompt,
            image_path=test_image_file,
            detected_objects=sample_sam_output
        )
    else:
        print(f"[Error] Test image not found at '{test_image_file}'.")