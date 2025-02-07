from accelerate import Accelerator

accelerator = Accelerator()
accelerator.load_state("checkpoint_dir")
