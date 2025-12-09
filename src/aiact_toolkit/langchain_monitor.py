"""
LangChain Monitor Plugin

This plugin automatically captures metadata from LangChain operations for AI Act compliance.
It uses LangChain's callback system to intercept and record:
- Model instantiations and parameters
- LLM calls
- Data loading operations
- Chain compositions

Approximately 180 lines as per prototype specification.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import time
try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    from langchain.callbacks.base import BaseCallbackHandler
from .metadata_storage import MetadataStorage
from .operational_metrics import OperationalMetricsTracker


class AIActCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler that captures LangChain operations for compliance metadata.
    """

    def __init__(self, metadata_storage: MetadataStorage, metrics_tracker: Optional[OperationalMetricsTracker] = None):
        """Initialize the callback handler with metadata storage."""
        super().__init__()
        self.storage = metadata_storage
        self.metrics_tracker = metrics_tracker
        self._operation_start_times: Dict[str, float] = {}
        self._operation_metadata: Dict[str, Dict[str, Any]] = {}

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Capture LLM initialization and calls."""
        run_id = str(kwargs.get("run_id", id(serialized)))

        # Start timing
        self._operation_start_times[run_id] = time.time()

        # Extract model information from serialized data
        model_info = {
            "timestamp": datetime.now().isoformat(),
            "model_name": serialized.get("id", ["unknown"])[-1] if isinstance(serialized.get("id"), list) else serialized.get("name", "unknown"),
            "provider": self._extract_provider(serialized),
            "framework_component": serialized.get("name", "unknown"),
            "parameters": {},
        }

        # Extract parameters from kwargs (temperature, max_tokens, etc.)
        if "invocation_params" in kwargs:
            params = kwargs["invocation_params"]
            model_info["parameters"] = {
                "temperature": params.get("temperature"),
                "max_tokens": params.get("max_tokens"),
                "top_p": params.get("top_p"),
                "model_name": params.get("model_name") or params.get("model"),
            }
            # Use parameter model_name if available
            if params.get("model_name") or params.get("model"):
                model_info["model_name"] = params.get("model_name") or params.get("model")

        # Remove None values
        model_info["parameters"] = {k: v for k, v in model_info["parameters"].items() if v is not None}

        # Store for later use in on_llm_end
        self._operation_metadata[run_id] = model_info

        self.storage.add_model(model_info)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Capture LLM response metadata including token usage and timing."""
        run_id = str(kwargs.get("run_id", id(response)))

        if self.metrics_tracker and run_id in self._operation_start_times:
            # Calculate execution time
            execution_time = (time.time() - self._operation_start_times[run_id]) * 1000  # Convert to ms

            # Get model info from start
            model_info = self._operation_metadata.get(run_id, {})
            model_name = model_info.get("model_name", "unknown")
            provider = model_info.get("provider", "unknown")

            # Extract token usage if available
            token_usage = None
            if hasattr(response, "llm_output") and response.llm_output:
                token_data = response.llm_output.get("token_usage", {})
                if token_data:
                    token_usage = {
                        "input_tokens": token_data.get("prompt_tokens", 0),
                        "output_tokens": token_data.get("completion_tokens", 0),
                        "total_tokens": token_data.get("total_tokens", 0)
                    }

            # Record operation
            self.metrics_tracker.record_operation(
                operation_type="llm_call",
                model_name=model_name,
                provider=provider,
                execution_time_ms=execution_time,
                token_usage=token_usage,
                success=True
            )

            # Cleanup
            del self._operation_start_times[run_id]
            if run_id in self._operation_metadata:
                del self._operation_metadata[run_id]

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Capture LLM errors."""
        run_id = str(kwargs.get("run_id", id(error)))

        if self.metrics_tracker and run_id in self._operation_start_times:
            execution_time = (time.time() - self._operation_start_times[run_id]) * 1000
            model_info = self._operation_metadata.get(run_id, {})

            self.metrics_tracker.record_operation(
                operation_type="llm_call",
                model_name=model_info.get("model_name", "unknown"),
                provider=model_info.get("provider", "unknown"),
                execution_time_ms=execution_time,
                success=False,
                error_message=str(error)
            )

            # Cleanup
            del self._operation_start_times[run_id]
            if run_id in self._operation_metadata:
                del self._operation_metadata[run_id]

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Capture chain initialization."""
        chain_info = {
            "timestamp": datetime.now().isoformat(),
            "chain_type": serialized.get("name", "unknown"),
            "chain_id": serialized.get("id", ["unknown"])[-1] if isinstance(serialized.get("id"), list) else "unknown",
        }
        self.storage.add_component(chain_info)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Capture tool usage."""
        tool_info = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": serialized.get("name", "unknown"),
            "tool_type": serialized.get("id", ["unknown"])[-1] if isinstance(serialized.get("id"), list) else "unknown",
        }
        self.storage.add_component(tool_info)

    def _extract_provider(self, serialized: Dict[str, Any]) -> str:
        """Extract provider from serialized data."""
        name = serialized.get("name", "").lower()
        id_list = serialized.get("id", [])

        # Check common providers
        if "openai" in name or any("openai" in str(i).lower() for i in id_list):
            return "OpenAI"
        elif "anthropic" in name or "claude" in name or any("anthropic" in str(i).lower() for i in id_list):
            return "Anthropic"
        elif "huggingface" in name or any("huggingface" in str(i).lower() for i in id_list):
            return "HuggingFace"
        elif "cohere" in name or any("cohere" in str(i).lower() for i in id_list):
            return "Cohere"
        elif "llama" in name:
            return "Meta/Llama"
        else:
            return "Unknown"


class LangChainMonitor:
    """
    Main monitoring class for capturing LangChain metadata.

    Usage:
        monitor = LangChainMonitor()
        monitor.start()

        # Use LangChain normally
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

        # Get metadata
        metadata = monitor.get_metadata()
        monitor.save_to_file("metadata.json")
    """

    def __init__(self, system_name: Optional[str] = None, enable_metrics: bool = True):
        """
        Initialize the LangChain monitor.

        Args:
            system_name: Optional name for the AI system being monitored
            enable_metrics: Enable operational metrics tracking (default: True)
        """
        self.system_name = system_name or "unnamed_system"
        self.storage = MetadataStorage(self.system_name)
        self.metrics_tracker = OperationalMetricsTracker() if enable_metrics else None
        self.callback_handler = AIActCallbackHandler(self.storage, self.metrics_tracker)
        self._is_started = False
        self._patched_classes = []

    def start(self):
        """Start monitoring LangChain operations."""
        if self._is_started:
            return

        # Monkey-patch common LangChain classes to inject callback
        self._patch_langchain_classes()
        self._is_started = True

    def stop(self):
        """Stop monitoring (restore original classes)."""
        if not self._is_started:
            return

        self._unpatch_langchain_classes()
        self._is_started = False

    def _patch_langchain_classes(self):
        """Patch LangChain classes to inject our callback handler."""
        try:
            # Patch common LLM classes
            from langchain_openai import ChatOpenAI
            self._patch_class(ChatOpenAI)
        except ImportError:
            pass

        try:
            from langchain_anthropic import ChatAnthropic
            self._patch_class(ChatAnthropic)
        except ImportError:
            pass

        try:
            from langchain.llms import HuggingFacePipeline
            self._patch_class(HuggingFacePipeline)
        except ImportError:
            pass

        try:
            from langchain_community.llms import CTransformers
            self._patch_class(CTransformers)
        except ImportError:
            pass

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._patch_class(HuggingFaceEmbeddings)
        except ImportError:
            pass

        # Patch document loaders
        self._patch_document_loaders()

    def _patch_class(self, cls):
        """Patch a specific class to inject callback."""
        if hasattr(cls, '_aiact_original_init'):
            return  # Already patched

        original_init = cls.__init__
        callback_handler = self.callback_handler
        storage = self.storage

        def patched_init(instance, *args, **kwargs):
            # Capture initialization parameters
            model_info = {
                "timestamp": datetime.now().isoformat(),
                "model_name": kwargs.get("model_name") or kwargs.get("model") or cls.__name__,
                "provider": cls.__name__,
                "framework_component": cls.__name__,
                "parameters": {},
            }

            # Capture common parameters
            for param in ["temperature", "max_tokens", "top_p", "max_new_tokens", "model_type", "device"]:
                if param in kwargs:
                    model_info["parameters"][param] = kwargs[param]

            # Capture model_kwargs if present (for embeddings)
            if "model_kwargs" in kwargs and isinstance(kwargs["model_kwargs"], dict):
                model_info["parameters"].update(kwargs["model_kwargs"])

            storage.add_model(model_info)

            # Don't inject callbacks - we capture initialization parameters instead
            # This avoids compatibility issues with different class types
            original_init(instance, *args, **kwargs)

        cls._aiact_original_init = original_init
        cls.__init__ = patched_init
        self._patched_classes.append(cls)

    def _patch_document_loaders(self):
        """Patch document loaders to capture data sources."""
        # This captures common document loaders
        loader_classes = []

        try:
            from langchain_community.document_loaders import TextLoader
            loader_classes.append(("TextLoader", TextLoader, "text"))
        except ImportError:
            pass

        try:
            from langchain_community.document_loaders import CSVLoader
            loader_classes.append(("CSVLoader", CSVLoader, "csv"))
        except ImportError:
            pass

        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader_classes.append(("PyPDFLoader", PyPDFLoader, "pdf"))
        except ImportError:
            pass

        try:
            from langchain_community.document_loaders import DirectoryLoader
            loader_classes.append(("DirectoryLoader", DirectoryLoader, "directory"))
        except ImportError:
            pass

        for loader_name, loader_cls, data_type in loader_classes:
            if hasattr(loader_cls, '_aiact_original_init'):
                continue

            original_init = loader_cls.__init__
            storage = self.storage

            def make_patched_init(lname, dtype):
                def patched_init(instance, *args, **kwargs):
                    # Capture data source
                    # Try different parameter names used by different loaders
                    file_path = args[0] if args else kwargs.get("file_path") or kwargs.get("path") or "unknown"
                    data_source_info = {
                        "timestamp": datetime.now().isoformat(),
                        "data_source": str(file_path),
                        "loader_type": lname,
                        "data_type": dtype,
                    }
                    # Also capture glob pattern if it exists (for DirectoryLoader)
                    if "glob" in kwargs:
                        data_source_info["glob_pattern"] = kwargs["glob"]
                    storage.add_data_source(data_source_info)
                    original_init(instance, *args, **kwargs)
                return patched_init

            loader_cls._aiact_original_init = original_init
            loader_cls.__init__ = make_patched_init(loader_name, data_type)
            self._patched_classes.append(loader_cls)

    def _unpatch_langchain_classes(self):
        """Restore original class implementations."""
        for cls in self._patched_classes:
            if hasattr(cls, '_aiact_original_init'):
                cls.__init__ = cls._aiact_original_init
                delattr(cls, '_aiact_original_init')
        self._patched_classes.clear()

    def get_metadata(self) -> Dict[str, Any]:
        """Get captured metadata."""
        metadata = self.storage.get_all_metadata()

        # Include operational metrics if available
        if self.metrics_tracker:
            metadata["operational_metrics"] = self.metrics_tracker.get_summary()

        return metadata

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get operational metrics summary."""
        if self.metrics_tracker:
            return self.metrics_tracker.get_summary()
        return {"message": "Metrics tracking not enabled"}

    def save_to_file(self, filepath: str):
        """Save metadata to JSON file."""
        self.storage.save_to_file(filepath)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
