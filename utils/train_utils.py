import itertools
from detectron2.solver.build import maybe_add_gradient_clipping
from typing import Set, Sequence, List, Any, Dict
import torch
from detectron2.engine import hooks
import detectron2.utils.comm as comm
from detectron2.evaluation.testing import flatten_results_dict


def build_optimizer(cfg, model):
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    gradient_accum = cfg.SOLVER.GRAD_STEP
    for key, value in model.named_parameters(recurse=True):
        if not value.requires_grad:
            continue
        # Avoid duplicating parameters
        if value in memo:
            continue
        memo.add(value)
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "backbone" in key:
            lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim
    
    def add_gradient_accumultion(optim):
        accum_steps = gradient_accum
        class GradAccumOptim(optim):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._iter_counter = 0
                
            def zero_grad(self):
                if self._iter_counter % accum_steps == 0 and self._iter_counter != 0:
                    super().zero_grad()
                self._iter_counter += 1 
        
        return GradAccumOptim if accum_steps > 1 else optim
                 

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = add_gradient_accumultion(maybe_add_full_model_gradient_clipping(torch.optim.SGD))(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "ADAMW":
        optimizer = add_gradient_accumultion(maybe_add_full_model_gradient_clipping(torch.optim.AdamW))(
            params, cfg.SOLVER.BASE_LR
        )
    elif optimizer_type == "ADAM":
        optimizer = add_gradient_accumultion(maybe_add_full_model_gradient_clipping(torch.optim.Adam))(
            params, cfg.SOLVER.BASE_LR
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    
    return optimizer



class EvalTestHook(hooks.HookBase):
    """
    Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function, test_function, eval_after_train=True):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still evaluate after the last iteration
                if `eval_after_train` is True).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
            eval_after_train (bool): whether to evaluate after the last iteration

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._eval_func = eval_function
        self._test_func = test_function
        self._eval_after_train = eval_after_train

    def _do_eval(self, test=False):
        
        if test:
            results = self._test_func()
        else:
            results = self._eval_func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._do_eval(test=False)

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        if self._eval_after_train and self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval(test=True)
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._eval_func
        del self._test_func