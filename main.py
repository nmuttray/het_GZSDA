from src.pyth.Wrapper import *

if __name__ == '__main__':
    # one run for all possible combinations of the primary motor cortex
    run_all = Wrapper("datasets/brain/brain_mouse_red_", "datasets/brain/brain_human_red_", "brain_",
                      ["scetm"], 2, dim=11, alpha=[0.01, 0.1, 1, 10, 100],
                      n_resample_source=300, n_resample_target=300)
    run_all.run_complete()



