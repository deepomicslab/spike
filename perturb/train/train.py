from pathlib import Path

import torch
import numpy as np
import pandas as pd
import random
from absl import logging
from perturb import losses
from perturb.utils.loaders import load
from perturb.train.summary import Logger
from perturb.data.utils import cast_loader_to_iterator
from tqdm import trange
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import normal
from sklearn.metrics import r2_score
import gzip
import pickle
import os


# ============================================
# 延迟导入模型特定的函数
# 避免加载不需要的模型依赖
# ============================================
def _lazy_import(module_name, attr_name):
    """延迟导入辅助函数"""
    module = __import__(f'perturb.models.{module_name}', fromlist=[attr_name])
    return getattr(module, attr_name)


# 缓存已导入的函数，避免重复导入
_imported_funcs = {}


def _get_model_func(module_name, func_name):
    """获取模型函数，支持延迟导入和缓存"""
    key = (module_name, func_name)
    if key not in _imported_funcs:
        _imported_funcs[key] = _lazy_import(module_name, func_name)
    return _imported_funcs[key]


# 便捷访问函数
def _compute_loss_f(*args, **kwargs):
    return _get_model_func('cellot', 'compute_loss_f')(*args, **kwargs)


def _compute_loss_g(*args, **kwargs):
    return _get_model_func('cellot', 'compute_loss_g')(*args, **kwargs)


def _compute_w2_distance(*args, **kwargs):
    return _get_model_func('cellot', 'compute_w2_distance')(*args, **kwargs)


def _compute_state_shift(*args, **kwargs):
    return _get_model_func('state', 'compute_state_shift')(*args, **kwargs)


def _compute_stack_shift(*args, **kwargs):
    return _get_model_func('stack', 'compute_stack_shift')(*args, **kwargs)


def _compute_ablation_loss(*args, **kwargs):
    return _get_model_func('ablation', 'compute_ablation_loss')(*args, **kwargs)


def _compute_scgpt_shift(*args, **kwargs):
    return _get_model_func('scgpt', 'compute_scgpt_shift')(*args, **kwargs)


def _compute_scgen_shift(*args, **kwargs):
    return _get_model_func('ae', 'compute_scgen_shift')(*args, **kwargs)


def _compute_prnet_loss(*args, **kwargs):
    return _get_model_func('PRnet', 'compute_prnet_loss')(*args, **kwargs)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_lr_scheduler(optim, config):
    if "scheduler" not in config:
        return None
    return torch.optim.lr_scheduler.StepLR(optim, **config.scheduler)


def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


def _cellot_expression_scale_stats(values):
    values = np.asarray(values, dtype=np.float64)
    abs_values = np.abs(values)
    return {
        "mean_abs": float(abs_values.mean()),
        "std": float(values.std()),
        "p99_abs": float(np.quantile(abs_values, 0.99)),
        "max_abs": float(abs_values.max()),
    }


def _cellot_scale_is_sane(target, transport):
    ratio_bounds = {
        "mean_abs": (0.2, 3.0),
        "std": (0.2, 3.0),
        "p99_abs": (0.2, 5.0),
        "max_abs": (0.2, 10.0),
    }
    eps = 1e-8
    target_stats = _cellot_expression_scale_stats(target)
    transport_stats = _cellot_expression_scale_stats(transport)
    ratio_stats = {
        key: (transport_stats[key] + eps) / (target_stats[key] + eps)
        for key in ratio_bounds
    }
    is_sane = all(
        lower <= ratio_stats[key] <= upper
        for key, (lower, upper) in ratio_bounds.items()
    )
    return is_sane, ratio_stats


def load_item_from_save(path, key, default):
    path = Path(path)
    if not path.exists():
        return default

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if key not in ckpt:
        logging.warning(f"'{key}' not found in ckpt: {str(path)}")
        return default

    return ckpt[key]


def get_training_seed(config, default=42):
    """Read an optional experiment seed from config while keeping legacy default."""
    return int(getattr(config, "seed", default))


def train_cellot(outdir, config):
    set_seed(get_training_seed(config))
    def state_dict(f, g, opts, **kwargs):
        state = {"g_state": g.state_dict(), "f_state": f.state_dict(),
                 "opt_g_state": opts.g.state_dict(), "opt_f_state": opts.f.state_dict()}
        state.update(kwargs)
        return state

    def evaluate():
        target = next(iterator_test_target)
        source = next(iterator_test_source)
        source.requires_grad_(True)
        transport = g.transport(source)
        transport = transport.detach()
        transport_np = transport.numpy()
        target_np = target.detach().numpy()
        scale_sane, scale_ratios = _cellot_scale_is_sane(target_np, transport_np)
        mean_corr = float(pd.Series(target_np.mean(0)).corr(pd.Series(transport_np.mean(0))))
        with torch.no_grad():
            gl = _compute_loss_g(f, g, source, transport).mean()
            fl = _compute_loss_f(f, g, source, target, transport).mean()
            dist = _compute_w2_distance(f, g, source, target, transport)
            mmd = losses.compute_scalar_mmd(
                target_np, transport_np)
        logger.log(
            "eval",
            gloss=gl.item(),
            floss=fl.item(),
            jloss=dist.item(),
            mmd=mmd,
            mean_corr=mean_corr,
            scale_sane=float(scale_sane),
            mean_abs_ratio=scale_ratios["mean_abs"],
            std_ratio=scale_ratios["std"],
            p99_abs_ratio=scale_ratios["p99_abs"],
            max_abs_ratio=scale_ratios["max_abs"],
            step=step)
        check_loss(gl, gl, dist)
        return mmd, scale_sane, mean_corr

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    (f, g), opts, loader = load(config, restore=cachedir / "last.pt")
    iterator = cast_loader_to_iterator(loader)

    n_iters = config.training['n_iters']
    step = load_item_from_save(cachedir / "last.pt", "step", 0)

    minmmd = load_item_from_save(cachedir / "model.pt", "minmmd", np.inf)
    best_mean_corr = load_item_from_save(cachedir / "model.pt", "best_mean_corr", -np.inf)
    mmd = minmmd
    if 'pair_batch_on' in config.training:
        keys = list(iterator.train.target.keys())
        test_keys = list(iterator.test.target.keys())
    else:
        keys = None
    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:
        if 'pair_batch_on' in config.training:
            assert keys is not None
            key = random.choice(keys)
            iterator_train_target = iterator.train.target[key]
            iterator_train_source = iterator.train.source[key]
            try:
                iterator_test_target = iterator.test.target[key]
                iterator_test_source = iterator.test.source[key]
            # in the iid mode of the ood setting,
            # train and test keys are not necessarily the same ...
            except KeyError:
                test_key = random.choice(test_keys)
                iterator_test_target = iterator.test.target[test_key]
                iterator_test_source = iterator.test.source[test_key]

        else:
            iterator_train_target = iterator.train.target
            iterator_train_source = iterator.train.source
            iterator_test_target = iterator.test.target
            iterator_test_source = iterator.test.source

        target = next(iterator_train_target)
        for _ in range(config.training['n_inner_iters']):
            source = next(iterator_train_source).requires_grad_(True)
            opts.g.zero_grad()
            gl = _compute_loss_g(f, g, source).mean()
            if not g.softplus_W_kernels and g.fnorm_penalty > 0:
                gl = gl + g.penalize_w()
            gl.backward()
            opts.g.step()
        source = next(iterator_train_source).requires_grad_(True)

        opts.f.zero_grad()
        fl = _compute_loss_f(f, g, source, target).mean()
        fl.backward()
        opts.f.step()
        check_loss(gl, fl)
        # print(f'gl loss {gl.item():.4f} fl loss {fl.item():.4f}')
        f.clamp_w()
        if step % config.training['logs_freq'] == 0:
            # log to logger object
            logger.log("train", gloss=gl.item(), floss=fl.item(), step=step)

        if step % config.training['eval_freq'] == 0:
            mmd, scale_sane, mean_corr = evaluate()
            better_corr = mean_corr > best_mean_corr + 1e-6
            tie_on_corr = abs(mean_corr - best_mean_corr) <= 1e-6
            better_mmd = mmd < minmmd
            if scale_sane and (better_corr or (tie_on_corr and better_mmd)):
                minmmd = mmd
                best_mean_corr = mean_corr
                torch.save(
                    state_dict(f, g, opts, step=step, minmmd=minmmd, best_mean_corr=best_mean_corr),
                    cachedir / "model.pt")

        if step % config.training['cache_freq'] == 0:
            torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")
            logger.flush()
    torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")
    logger.flush()

    # Ensure model.pt always exists - save as best model if it doesn't exist
    # or save the final model if no improvement was ever found
    if not (cachedir / "model.pt").exists():
        logging.info("No model.pt found, saving final model as best model")
        torch.save(
            state_dict(f, g, opts, step=step, minmmd=mmd, best_mean_corr=best_mean_corr),
            cachedir / "model.pt")
    return


def train_auto_encoder(outdir, config):
    set_seed(42)
    def state_dict(model, optim, **kwargs):
        state = {"model_state": model.state_dict(),
                 "optim_state": optim.state_dict()}
        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means
        state.update(kwargs)
        return state

    def evaluate(vinputs):
        with torch.no_grad():
            loss, comps, _ = model(vinputs)
            loss = loss.mean()
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}
            check_loss(loss)
            logger.log("eval", loss=loss.item(), step=step, **comps)
        return loss

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, restore=cachedir / "last.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iterator = cast_loader_to_iterator(loader)
    scheduler = load_lr_scheduler(optim, config)

    n_iters = config.training['n_iters']
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step

    best_eval_loss = load_item_from_save(
        cachedir / "model.pt", "best_eval_loss", np.inf)

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:
        model.train()
        inputs = next(iterator.train).to(device)
        optim.zero_grad()
        loss, comps, _ = model(inputs)
        loss = loss.mean()
        comps = {k: v.mean().item() for k, v in comps._asdict().items()}
        loss.backward()
        optim.step()
        check_loss(loss)

        if step % config.training['logs_freq'] == 0:
            logger.log("train", loss=loss.item(), step=step, **comps)

        if step % config.training['eval_freq'] == 0:
            model.eval()
            eval_loss = evaluate(next(iterator.test).to(device))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)
                torch.save(sd, cachedir / "model.pt")
            print(f'loss {eval_loss.item():.4f}')

        if step % config.training['cache_freq'] == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")
            logger.flush()

        if scheduler is not None:
            scheduler.step()

    if config.model['name'] == "scgen" and config.compute_scgen_shift == True:
        labels = loader.train.dataset.adata.obs[config.data['condition']]
        _compute_scgen_shift(model, loader.train.dataset, labels=labels)
    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")
    logger.flush()

    # Ensure model.pt always exists
    if not (cachedir / "model.pt").exists():
        logging.info("No model.pt found, saving final model as best model")
        torch.save(state_dict(model, optim, step=step, eval_loss=eval_loss if 'eval_loss' in dir() else np.inf),
                   cachedir / "model.pt")


def train_prnet(outdir, config):
    set_seed(42)
    def state_dict(model, optim, **kwargs):
        state = {"model_state": model.state_dict(),
                 "optim_state": optim.state_dict()}
        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means
        state.update(kwargs)
        return state

    def evaluate(vinputs):
        (control, target), encode_label = vinputs['features'], vinputs['label']
        control = control.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)
        encode_label = encode_label.to(device, dtype=torch.float32)
        b_size = control.size(0)
        noise = make_noise(b_size, 10)
        with torch.no_grad():
            gene_reconstructions = model(control, encode_label, noise).detach()
            dim = gene_reconstructions.size(1) // 2
            gene_means = gene_reconstructions[:, :dim]
            gene_vars = gene_reconstructions[:, dim:]
            gene_vars = F.softplus(gene_vars)
            dist = normal.Normal(
                torch.clamp(torch.Tensor(gene_means), min=1e-3, max=1e3),
                torch.clamp(torch.Tensor(gene_vars.sqrt()), min=1e-3, max=1e3))
            nb_sample = dist.sample().cpu().numpy()
            yp_m = nb_sample.mean(0)
            y_true = target.cpu().numpy()
            yt_m = y_true.mean(axis=0)
            r2_score_mean = r2_score(yt_m, yp_m)
            print(f'eval R2 {r2_score_mean}')
            loss = _compute_prnet_loss(gene_means, target, gene_vars)
            check_loss(loss)
            logger.log("eval", loss=loss.item(), step=step)
        return loss

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, restore=cachedir / "last.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iterator = cast_loader_to_iterator(loader)
    scheduler = load_lr_scheduler(optim, config)

    n_iters = config.training['n_iters']
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step

    best_eval_loss = load_item_from_save(
        cachedir / "model.pt", "best_eval_loss", np.inf)

    ticker = trange(step, n_iters, initial=step, total=n_iters)

    def make_noise(batch_size, shape, volatile=False):
        tensor = torch.randn(batch_size, shape)
        noise = Variable(tensor, volatile)
        noise = noise.to(device, dtype=torch.float32)
        return noise

    for step in ticker:
        model.train()
        optim.zero_grad()
        inputs = next(iterator.train)
        (control, target), encode_label = inputs['features'], inputs['label']
        control = control.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)
        encode_label = encode_label.to(device, dtype=torch.float32)
        b_size = control.size(0)
        noise = make_noise(b_size, 10)
        gene_reconstructions = model(control, encode_label, noise)
        dim = gene_reconstructions.size(1) // 2
        gene_means = gene_reconstructions[:, :dim]
        gene_vars = gene_reconstructions[:, dim:]
        gene_vars = F.softplus(gene_vars)
        loss = _compute_prnet_loss(gene_means, target, gene_vars)
        loss.backward()
        optim.step()
        check_loss(loss)

        if step % config.training['logs_freq'] == 0:
            logger.log("train", loss=loss.item(), step=step)

        if step % config.training['eval_freq'] == 0:
            model.eval()
            eval_loss = evaluate(next(iterator.test))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)
                torch.save(sd, cachedir / "model.pt")

        if step % config.training['cache_freq'] == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")
            logger.flush()

        if scheduler is not None:
            scheduler.step()

    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")
    logger.flush()

    # Ensure model.pt always exists
    if not (cachedir / "model.pt").exists():
        logging.info("No model.pt found, saving final model as best model")
        torch.save(state_dict(model, optim, step=step, eval_loss=eval_loss if 'eval_loss' in dir() else np.inf),
                   cachedir / "model.pt")


def train_scgpt(outdir, config):
    set_seed(42)
    def state_dict(model, optim, **kwargs):
        state = {"model_state": model.state_dict(),
                 "optim_state": optim.state_dict()}
        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means
        state.update(kwargs)
        return state

    def evaluate(vfeature, vinputs):
        with torch.no_grad():
            loss, comps, _ = model(vfeature, vinputs)
            loss = loss.mean()
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}
            check_loss(loss)
            # logger.log("eval", loss=loss.item(), **comps)
            print(f"eval loss {loss.item()}")
        return loss

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, restore=cachedir / "last.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iterator = cast_loader_to_iterator(loader)
    scheduler = load_lr_scheduler(optim, config)

    n_iters = config.training['n_iters']
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step

    best_eval_loss = load_item_from_save(
        cachedir / "model.pt", "best_eval_loss", np.inf)

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:
        model.train()
        inputs_emb, inputs = next(iterator.train)
        optim.zero_grad()
        loss, comps, _ = model(inputs_emb.to(device, dtype=torch.float32), inputs.to(device, dtype=torch.float32))
        loss = loss.mean()
        comps = {k: v.mean().item() for k, v in comps._asdict().items()}
        loss.backward()
        optim.step()
        check_loss(loss)
        if step % config.training['logs_freq'] == 0:
            # logger.log("train", loss=loss.item(), **comps)
            print(f'step: {step} loss{loss.item()}')

        if step % config.training['eval_freq'] == 0:
            model.eval()
            feature, inputs = next(iterator.test)
            eval_loss = evaluate(feature.to(device), inputs.to(device))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)
                torch.save(sd, cachedir / "model.pt")
            print(f'loss {eval_loss.item():.4f}')
        if step % config.training['cache_freq'] == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")
            logger.flush()

        if scheduler is not None:
            scheduler.step()

    if config.model['name'] == "scgpt" and config.compute_scgpt_shift == True:
        labels = loader.train.dataset.adata.obs[config.data['condition']]
        _compute_scgpt_shift(model, loader.train.dataset, labels=labels)
    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")
    logger.flush()

    # Ensure model.pt always exists
    if not (cachedir / "model.pt").exists():
        logging.info("No model.pt found, saving final model as best model")
        torch.save(state_dict(model, optim, step=step, eval_loss=eval_loss if 'eval_loss' in dir() else np.inf),
                   cachedir / "model.pt")


def  train_state(outdir, config):
    set_seed(42)
    def state_dict(model, optim, **kwargs):
        state = {"model_state": model.state_dict(),
                 "optim_state": optim.state_dict()}
        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means
        state.update(kwargs)
        return state

    def evaluate(vfeature, vinputs):
        with torch.no_grad():
            loss, comps, _ = model(vfeature, vinputs)
            loss = loss.mean()
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}
            check_loss(loss)
            # logger.log("eval", loss=loss.item(), **comps)
            print(f"eval loss {loss.item()}")
        return loss

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, restore=cachedir / "last.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iterator = cast_loader_to_iterator(loader)
    scheduler = load_lr_scheduler(optim, config)

    n_iters = config.training['n_iters']
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step

    best_eval_loss = load_item_from_save(
        cachedir / "model.pt", "best_eval_loss", np.inf)

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:
        model.train()
        inputs_emb, inputs = next(iterator.train)
        optim.zero_grad()
        loss, comps, _ = model(inputs_emb.to(device, dtype=torch.float32), inputs.to(device, dtype=torch.float32))
        loss = loss.mean()
        comps = {k: v.mean().item() for k, v in comps._asdict().items()}
        loss.backward()
        optim.step()
        check_loss(loss)
        if step % config.training['logs_freq'] == 0:
            # logger.log("train", loss=loss.item(), **comps)
            print(f'step: {step} loss{loss.item()}')

        if step % config.training['eval_freq'] == 0:
            model.eval()
            feature, inputs = next(iterator.test)
            eval_loss = evaluate(feature.to(device), inputs.to(device))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)
                torch.save(sd, cachedir / "model.pt")
            print(f'loss {eval_loss.item():.4f}')
        if step % config.training['cache_freq'] == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")
            logger.flush()

        if scheduler is not None:
            scheduler.step()

    if config.model['name'] == "state" and config.compute_state_shift == True:
        labels = loader.train.dataset.adata.obs[config.data['condition']]
        _compute_state_shift(model, loader.train.dataset, labels=labels)
    if config.model['name'] == "stack" and config.compute_stack_shift == True:
        labels = loader.train.dataset.adata.obs[config.data['condition']]
        _compute_stack_shift(model, loader.train.dataset, labels=labels)
    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")
    logger.flush()

    # Ensure model.pt always exists
    if not (cachedir / "model.pt").exists():
        logging.info("No model.pt found, saving final model as best model")
        torch.save(state_dict(model, optim, step=step, eval_loss=eval_loss if 'eval_loss' in dir() else np.inf),
                   cachedir / "model.pt")

def save_attention_score(DGI_model, cachedir):
    X_attention_index = DGI_model.encoder_net.attention_scores_mine[0]
    X_attention_index = X_attention_index.cpu().detach().numpy()
    X_attention_score_normalized_l1 = DGI_model.encoder_net.attention_scores_mine_l1[1]
    X_attention_score_normalized_l1 = X_attention_score_normalized_l1.cpu().detach().numpy()
    X_attention_score_unnormalized_l1 = DGI_model.encoder_net.attention_scores_mine_unnormalized_l1
    X_attention_score_unnormalized_l1 = X_attention_score_unnormalized_l1.cpu().detach().numpy()
    X_attention_score_normalized = DGI_model.encoder_net.attention_scores_mine[1]
    X_attention_score_normalized = X_attention_score_normalized.cpu().detach().numpy()
    X_attention_score_unnormalized = DGI_model.encoder_net.attention_scores_mine_unnormalized
    X_attention_score_unnormalized = X_attention_score_unnormalized.cpu().detach().numpy()
    X_attention_bundle = [X_attention_index, X_attention_score_normalized_l1, X_attention_score_unnormalized,
                          X_attention_score_unnormalized_l1, X_attention_score_normalized]
    X_attention_filename = str(cachedir) + '/attention.npy'
    with gzip.open(X_attention_filename, 'wb') as fp:
        pickle.dump(X_attention_bundle, fp)


def train_CellNest(outdir, config):
    def state_dict(model, optim, **kwargs):
        state = {"model_state": model.state_dict(), "optim_state": optim.state_dict()}
        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means
        state.update(kwargs)
        return state

    def evaluate(test_data):
        with torch.no_grad():
            outputs = model(test_data.to(device))
            loss = model.loss(outputs)
            check_loss(loss)
            logger.log("eval", loss=loss.item(), step=step)
        return loss

    set_seed(42)
    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, restore=cachedir / "last.pt")
    scheduler = load_lr_scheduler(optim, config)
    n_iters = config.training['n_iters']
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step
    best_eval_loss = load_item_from_save(cachedir / "model.pt", "best_eval_loss", np.inf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:
        if config.data['mode'] == 'source':
            train_data = loader.train.source.dataset.graph
            test_data = loader.test.source.dataset.graph
        else:
            train_data = loader.train.target.dataset.graph
            test_data = loader.test.target.dataset.graph
        model.train()
        optim.zero_grad()
        outputs = model(train_data.to(device))
        loss = model.loss(outputs)
        loss.backward()
        optim.step()
        check_loss(loss)
        ticker.set_postfix(loss=f'{loss.item():.4f}')
        if step % config.training['logs_freq'] == 0:
            logger.log("train", loss=loss.item(), step=step)
        if step % config.training['eval_freq'] == 0:
            model.eval()
            eval_loss = evaluate(test_data)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)
                torch.save(sd, cachedir / "model.pt")
                save_attention_score(model.dgi, cachedir)
            print(f'\neval loss {eval_loss.item():.4f}')
        if step % config.training['cache_freq'] == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")
            logger.flush()

        if scheduler is not None:
            scheduler.step()

    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")
    logger.flush()

    # Ensure model.pt always exists
    if not (cachedir / "model.pt").exists():
        logging.info("No model.pt found, saving final model as best model")
        torch.save(state_dict(model, optim, step=step, eval_loss=eval_loss if 'eval_loss' in dir() else np.inf),
                   cachedir / "model.pt")


def train_ablation(outdir, config):
    """Train ablation models (single_decoder, standard_gat, no_edge)"""
    def state_dict(model, optim, **kwargs):
        state = {"model_state": model.state_dict(), "optim_state": optim.state_dict()}
        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means
        state.update(kwargs)
        return state

    def evaluate(test):
        source = test.source.dataset
        target = test.target.dataset
        with torch.no_grad():
            source_recon, target_recon = model(source.graph.to(device))
            loss_out = _compute_ablation_loss(source_recon, source, target_recon, target)
            check_loss(loss_out.total)
            logger.log("eval", loss=loss_out.total.detach().item(), step=step)
        return loss_out.total

    set_seed(42)
    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, restore=cachedir / "last.pt")
    scheduler = load_lr_scheduler(optim, config)
    n_iters = config.training['n_iters']
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step
    best_eval_loss = load_item_from_save(cachedir / "model.pt", "best_eval_loss", np.inf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:
        source = loader.train.source.dataset
        target = loader.train.target.dataset
        model.train()
        optim.zero_grad()
        source_recon, target_recon = model(source.graph.to(device))
        loss_out = _compute_ablation_loss(source_recon, source, target_recon, target)
        loss_out.total.backward()
        optim.step()
        check_loss(loss_out.total)
        ticker.set_postfix(loss=f'{loss_out.total.detach().item():.2f}', recon=f'{loss_out.recon.detach().item():.2f}', align=f'{loss_out.align.detach().item():.2f}')
        if step % config.training['logs_freq'] == 0:
            logger.log("train", loss=loss_out.total.detach().item(), step=step)
        if step % config.training['eval_freq'] == 0:
            model.eval()
            eval_loss = evaluate(loader.test)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)
                torch.save(sd, cachedir / "model.pt")
                save_attention_score(model, cachedir)
            print(f'\neval loss {eval_loss.item():.4f}')
        if step % config.training['cache_freq'] == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")
            logger.flush()

        if scheduler is not None:
            scheduler.step()

    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")
    logger.flush()

    # Ensure model.pt always exists
    if not (cachedir / "model.pt").exists():
        logging.info("No model.pt found, saving final model as best model")
        torch.save(state_dict(model, optim, step=step, eval_loss=eval_loss if 'eval_loss' in dir() else np.inf),
                   cachedir / "model.pt")


def train_stack(outdir, config):
    """Train stack model using pretrained embeddings."""
    set_seed(42)

    def state_dict(model, optim, **kwargs):
        state = {"model_state": model.state_dict(), "optim_state": optim.state_dict()}
        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means
        state.update(kwargs)
        return state

    def evaluate(vfeature, vinputs):
        with torch.no_grad():
            loss, comps, _ = model(vfeature, vinputs)
            loss = loss.mean()
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}
            check_loss(loss)
            logger.log("eval", loss=loss.item(), step=step, **comps)
        return loss

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, restore=cachedir / "last.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iterator = cast_loader_to_iterator(loader)
    scheduler = load_lr_scheduler(optim, config)

    n_iters = config.training['n_iters']
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step

    best_eval_loss = load_item_from_save(
        cachedir / "model.pt", "best_eval_loss", np.inf)

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:
        model.train()
        features, inputs = next(iterator.train)
        features = features.to(device)
        inputs = inputs.to(device)
        optim.zero_grad()
        loss, comps, _ = model(features, inputs)
        loss = loss.mean()
        comps = {k: v.mean().item() for k, v in comps._asdict().items()}
        loss.backward()
        optim.step()
        check_loss(loss)

        if step % config.training['logs_freq'] == 0:
            logger.log("train", loss=loss.item(), step=step, **comps)

        if step % config.training['eval_freq'] == 0:
            model.eval()
            vfeatures, vinputs = next(iterator.test)
            eval_loss = evaluate(vfeatures.to(device), vinputs.to(device))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)
                torch.save(sd, cachedir / "model.pt")
            print(f'loss {eval_loss.item():.4f}')

        if step % config.training['cache_freq'] == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")
            logger.flush()

        if scheduler is not None:
            scheduler.step()

    # Compute stack shift for perturbation prediction
    if getattr(config, 'compute_stack_shift', False):
        labels = loader.train.dataset.adata.obs[config.data['condition']]
        _compute_stack_shift(model, loader.train.dataset, labels=labels)

    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")
    logger.flush()

    # Ensure model.pt always exists
    if not (cachedir / "model.pt").exists():
        logging.info("No model.pt found, saving final model as best model")
        torch.save(state_dict(model, optim, step=step, eval_loss=eval_loss if 'eval_loss' in dir() else np.inf),
                   cachedir / "model.pt")
