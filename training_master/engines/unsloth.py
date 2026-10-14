"""Direct text causal-LM SFT with LoRA/QLoRA, following upstream references."""
from __future__ import annotations

import json
import importlib.metadata
from pathlib import Path

from ..artifacts import RunPaths, write_json
from ..hardware import HardwareProfile
from ..job import JobSpec
from ..bootstrap import UNSLOTH_VERSION
from .base import TrainingEngine, installed, materialize_dataset


class UnslothEngine(TrainingEngine):
    name = "unsloth"

    def available(self) -> bool:
        try:
            return all(installed(name) for name in ("unsloth", "trl", "datasets", "torch")) and (
                importlib.metadata.version("unsloth") == UNSLOTH_VERSION
                and importlib.metadata.version("trl") == "0.22.2"
                and importlib.metadata.version("transformers") == "4.57.6"
            )
        except importlib.metadata.PackageNotFoundError:
            return False

    def unsupported_fields(self, job: JobSpec) -> list[str]:
        errors = []
        if job.engine.backend is not None:
            errors.append("engine.backend: this option selects a Soup backend; omit it for the direct Unsloth engine")
        if job.task.type != "sft":
            errors.append(f"task.type={job.task.type}: direct engine supports text SFT only")
        if job.training.stream_layers:
            errors.append("training.stream_layers: use Soup's transformers backend")
        if not job.training.lora.enabled:
            errors.append("training.lora.enabled=false: direct engine supports LoRA/QLoRA only")
        if job.dataset.format != "plaintext" and job.dataset.text_field != "text":
            errors.append("dataset.text_field: applies only to plaintext datasets")
        return errors

    def supports(self, job: JobSpec, hardware: HardwareProfile) -> bool:
        return super().supports(job, hardware) and hardware.cuda_available

    def validate(self, job: JobSpec, hardware: HardwareProfile) -> None:
        super().validate(job, hardware)
        if not hardware.cuda_available:
            raise ValueError("Direct Unsloth requires working CUDA; CPU planning can use a recorded hardware profile")

    def dry_run(self, job: JobSpec, hardware: HardwareProfile, paths: RunPaths | None = None) -> dict:
        plan = super().dry_run(job, hardware, paths)
        # Conservative batch_size=auto resolves to 1; record it explicitly.
        plan["batch_size"] = 1 if job.training.batch_size == "auto" else job.training.batch_size
        plan["max_steps"] = job.training.max_steps
        plan["framework_preflight"] = False
        return plan

    def run(self, job: JobSpec, hardware: HardwareProfile, paths: RunPaths, *, resume: Path | None = None, on_checkpoint=None) -> dict:
        self.validate(job, hardware)
        # Import Unsloth before transformers/TRL, as upstream notebooks require.
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer

        t = job.training
        data_path = materialize_dataset(job, paths)
        rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=job.model.id, revision=job.model.revision,
            max_seq_length=t.max_length, dtype=None,
            load_in_4bit=t.quantization == "4bit",
        )
        model = FastLanguageModel.get_peft_model(
            model, r=t.lora.r, target_modules=t.lora.target_modules,
            lora_alpha=t.lora.alpha, lora_dropout=t.lora.dropout, bias="none",
            use_gradient_checkpointing="unsloth" if t.gradient_checkpointing else False,
            random_state=t.seed,
        )
        formatted = []
        for row in rows:
            if job.dataset.format == "plaintext":
                text = row["text"]
            else:
                if job.dataset.format == "alpaca":
                    prompt = row["instruction"] + ("\n" + row["input"] if row["input"] else "")
                    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": row["output"]}]
                else:
                    messages = row["messages"]
                if not tokenizer.chat_template:
                    raise ValueError("Chat/alpaca SFT requires a model tokenizer with a chat template")
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted.append({"text": text})

        class CheckpointHook(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                if on_checkpoint and state.is_world_process_zero:
                    on_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}")
                return control

        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=Dataset.from_list(formatted),
            args=SFTConfig(
                output_dir=str(paths.trainer), dataset_text_field="text", max_length=t.max_length,
                num_train_epochs=t.epochs, max_steps=t.max_steps or -1,
                per_device_train_batch_size=1 if t.batch_size == "auto" else t.batch_size,
                gradient_accumulation_steps=t.gradient_accumulation_steps,
                gradient_checkpointing=t.gradient_checkpointing,
                learning_rate=t.learning_rate, seed=t.seed, data_seed=t.seed,
                fp16=not hardware.bf16_supported, bf16=hardware.bf16_supported,
                save_strategy="steps", save_steps=job.artifacts.checkpoint_interval,
                logging_steps=1, report_to="none", packing=False,
            ), callbacks=[CheckpointHook()],
        )
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        model.save_pretrained(str(paths.final))
        tokenizer.save_pretrained(str(paths.final))
        trainer.save_state()
        trainer_state = paths.trainer / "trainer_state.json"
        if trainer_state.exists():
            import shutil
            shutil.copy2(trainer_state, paths.root / "metrics" / "trainer_state.json")
        write_json(paths.root / "metrics" / "train.json", result.metrics)
        return {"output": str(paths.final), "engine": self.name, "metrics": result.metrics}
