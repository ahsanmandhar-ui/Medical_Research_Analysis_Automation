"""TimesFM 1.0-200M CPU-only zero-shot forecast v2 (bfloat16 enforced).
- Model: google/timesfm-1.0-200m-pytorch (200M, torch) via timesfm==1.3.0 v1 loader
- CPU-only, per_core_batch_size=1, mmap load, bfloat16 weights + inputs
- Context 12 (2015-2026), horizon 5 (2027-2031), freq Y/low=2
- Memory budget <2GB, tracked via psutil
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import time
import types
import logging
import psutil
import numpy as np
import pandas as pd
import torch

def flush_print(*a, **k):
    print(*a, **k, flush=True)

proc = psutil.Process()
def mem_gb():
    return proc.memory_info().rss / (1024**3)
def log_mem(tag):
    rss = mem_gb()
    vm = psutil.virtual_memory()
    flush_print(f"[MEM] {tag}: RSS={rss:.3f} GB | avail={vm.available/(1024**3):.3f}/{vm.total/(1024**3):.3f} GB")
    return rss

flush_print("="*70)
flush_print("TimesFM 1.0-200M CPU FORECAST — bfloat16 enforced, <2GB")
flush_print("="*70)
flush_print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} (CPU-only required)")
torch.set_num_threads(2)
log_mem("startup")

# Precision probe
try:
    torch.randn(4,4,dtype=torch.bfloat16) @ torch.randn(4,4,dtype=torch.bfloat16)
    bf16_ok = True
except Exception as e:
    bf16_ok = False
    flush_print(f"bf16 probe failed: {e}")
flush_print(f"bfloat16 CPU compute available: {bf16_ok}")

# Data
hist = pd.read_csv("/home/user/meta_analysis_publications_2015_2026.csv")
flush_print(hist.to_string(index=False))
pubs = hist["Meta_Analysis_Publications"].to_numpy(dtype=np.float32)
ai_hist = hist["AI_Automation_Impact_Factor"].to_numpy(dtype=np.float32)
CONTEXT_LEN, HORIZON = 12, 5
FORECAST_YEARS = [2027,2028,2029,2030,2031]
assert len(pubs)==12

# Patch torch.load for mmap
_orig_load = torch.load
def _mmap_load(f, map_location=None, *a, **kw):
    if map_location is None: map_location="cpu"
    kw.setdefault("mmap", True); kw.setdefault("weights_only", True)
    return _orig_load(f, map_location=map_location, *a, **kw)
torch.load = _mmap_load

import timesfm.timesfm_base as tfm_base
from timesfm.timesfm_torch import TimesFmTorch

hparams = tfm_base.TimesFmHparams(
    context_len=512, horizon_len=HORIZON,
    input_patch_len=32, output_patch_len=128,
    num_layers=20, model_dims=1280, num_heads=16,
    per_core_batch_size=1, backend="cpu",
    quantiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    use_positional_embedding=True, point_forecast_mode="mean",
)
checkpoint = tfm_base.TimesFmCheckpoint(
    huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
    local_dir="/home/user/tmp/hf_timesfm_1p0_200m",
)
log_mem("before load")
t0=time.time()
tfm = TimesFmTorch(hparams=hparams, checkpoint=checkpoint)
torch.load = _orig_load
flush_print(f"Loaded in {time.time()-t0:.1f}s")
log_mem("after load fp32")
n_params = sum(p.numel() for p in tfm._model.parameters())
param_gb = sum(p.numel()*p.element_size() for p in tfm._model.parameters())/(1024**3)
flush_print(f"Params {n_params/1e6:.1f}M | weights {param_gb:.3f} GB fp32 | device {tfm._device}")

# Enforce bfloat16
active_dtype = torch.float32
if bf16_ok:
    try:
        flush_print("Converting weights to bfloat16 ...")
        gc.collect()
        tfm._model.to(torch.bfloat16)
        param_gb16 = sum(p.numel()*p.element_size() for p in tfm._model.parameters())/(1024**3)
        log_mem("after bf16 convert")
        flush_print(f"Weights now {param_gb16:.3f} GB bfloat16 (saved {param_gb-param_gb16:.3f} GB)")
        # Patch _forecast to cast inputs+padding to model dtype (bf16), outputs back to fp32
        def _bf16_forecast(self, inputs, freq=None, window_size=None, forecast_context_len=None, return_forecast_on_context=False):
            if self._model is None: raise ValueError("Checkpoint not loaded")
            if forecast_context_len is None: forecast_context_len=self.context_len
            inputs=[np.array(ts)[-forecast_context_len:] for ts in inputs]
            if window_size is not None:
                ni=[]
                for ts in inputs: ni.extend(tfm_base.moving_average(ts, window_size))
                inputs=ni
            if freq is None:
                logging.info("No freq, default high(0)")
                freq=[0]*len(inputs)
            input_ts, input_padding, inp_freq, pmap_pad = self._preprocess(inputs, freq)
            try: mdtype=next(self._model.parameters()).dtype
            except: mdtype=torch.float32
            with torch.no_grad():
                mo=[]; fo=[]
                for i in range(input_ts.shape[0]//self.global_batch_size):
                    a=input_ts[i*self.global_batch_size:(i+1)*self.global_batch_size]
                    b=input_padding[i*self.global_batch_size:(i+1)*self.global_batch_size]
                    c=inp_freq[i*self.global_batch_size:(i+1)*self.global_batch_size,:]
                    t_a=torch.Tensor(a).to(self._device).to(mdtype)
                    t_b=torch.Tensor(b).to(self._device).to(mdtype)
                    t_c=torch.LongTensor(c).to(self._device)
                    mean_o, full_o=self._model.decode(input_ts=t_a, paddings=t_b, freq=t_c,
                        horizon_len=self.horizon_len, output_patch_len=self.output_patch_len,
                        return_forecast_on_context=True)
                    if not return_forecast_on_context:
                        mean_o=mean_o[:,self._horizon_start:,...]; full_o=full_o[:,self._horizon_start:,...]
                    mo.append(mean_o.detach().float().numpy()); fo.append(full_o.detach().float().numpy())
            mo=np.concatenate(mo,axis=0); fo=np.concatenate(fo,axis=0)
            if pmap_pad>0: mo=mo[:-pmap_pad,...]; fo=fo[:-pmap_pad,...]
            if window_size is not None: mo=mo[0::2,...]+mo[1::2,...]; fo=fo[0::2,...]+fo[1::2,...]
            return mo, fo
        tfm._forecast = types.MethodType(_bf16_forecast, tfm)
        flush_print("Patched _forecast for bfloat16 (inputs+padding cast to bf16).")
        active_dtype=torch.bfloat16
    except Exception as e:
        flush_print(f"bf16 enforce failed ({e}), staying fp32")
        try: tfm._model.to(torch.float32)
        except: pass
        active_dtype=torch.float32
else:
    flush_print("bf16 unavailable, using fp32")

flush_print(f"Active dtype: {active_dtype}")
rss=log_mem("pre-inference")
assert rss<2.0, "Over 2GB!"
flush_print("Memory check PASSED (<2GB)")

# Inference
tfm._model.eval()
flush_print("Running zero-shot forecast (context=12, horizon=5, freq=2/Y) ...")
t1=time.time()
mean_fcst, full_fcst = tfm.forecast(inputs=[pubs], freq=[2], forecast_context_len=CONTEXT_LEN)
flush_print(f"Inference {time.time()-t1:.1f}s | mean{mean_fcst.shape} full{full_fcst.shape}")
log_mem("post-inference")

mean5=mean_fcst[0,:HORIZON]
q10=full_fcst[0,:HORIZON,1]
q50=full_fcst[0,:HORIZON,5]
q90=full_fcst[0,:HORIZON,9]

fcst_df=pd.DataFrame({
    "Year":FORECAST_YEARS,
    "Point_Forecast_Mean":np.round(mean5,1),
    "Q10_Worst_Case":np.round(q10,1),
    "Median_Q50":np.round(q50,1),
    "Q90_Best_Case":np.round(q90,1),
    "AI_Factor_Projected":[0.92,0.95,0.97,0.98,0.99],
})
fcst_df.to_csv("/home/user/meta_analysis_forecast_2027_2031.csv",index=False)
flush_print("Saved /home/user/meta_analysis_forecast_2027_2031.csv")

md=["| Year | Point Forecast (mean) | 10th pct (worst-case decline) | 90th pct (best-case stability) |",
    "|------|----------------------|-------------------------------|--------------------------------|"]
for _,r in fcst_df.iterrows():
    md.append(f"| {int(r['Year'])} | {r['Point_Forecast_Mean']:,.1f} | {r['Q10_Worst_Case']:,.1f} | {r['Q90_Best_Case']:,.1f} |")
md_text="\n".join(md)
flush_print("\n"+md_text+"\n")
with open("/home/user/forecast_table.md","w") as f:
    f.write("# TimesFM 1.0-200M Zero-Shot Forecast: Meta-Analysis Publications 2027–2031\n\n")
    f.write(f"Model: google/timesfm-1.0-200m-pytorch (200M, CPU, dtype={active_dtype})\n\n")
    f.write("Context: 12 pts (2015–2026) | Horizon: 5y | Freq: Y (low=2) | Batch: 1 | mmap load\n\n")
    f.write(md_text+"\n\n")
    f.write("_Note: TimesFM-1.0 quantile heads are experimental/uncalibrated (Google docs); q10/q90 are indicative._\n")

last_hist=float(pubs[-1]); last_fcst=float(mean5[-1])
chg=100*(last_fcst-last_hist)/last_hist
flush_print(f"2026={last_hist:.0f} 2031 mean={last_fcst:.0f} change {chg:+.2f}%")
flush_print(f"q10 2031 {float(q10[-1]):.0f} ({100*(float(q10[-1])-last_hist)/last_hist:+.2f}%) | q90 2031 {float(q90[-1]):.0f} ({100*(float(q90[-1])-last_hist)/last_hist:+.2f}%)")

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
years_hist=hist["Year"].to_numpy()
fig,ax1=plt.subplots(figsize=(11,6.5))
ax1.plot(years_hist,pubs,marker="o",linewidth=2.5,color="#1f77b4",label="Historical publications",zorder=3)
ax1.plot(FORECAST_YEARS,mean5,marker="s",linewidth=2.5,linestyle="--",color="#d62728",label=f"TimesFM point forecast (mean, {active_dtype})",zorder=3)
ax1.fill_between(FORECAST_YEARS,q10,q90,color="#d62728",alpha=0.15,label="Uncertainty band (q10–q90, experimental)")
ax1.plot(FORECAST_YEARS,q10,linestyle=":",linewidth=1.5,color="#7f7f7f",label="q10 worst-case decline")
ax1.plot(FORECAST_YEARS,q90,linestyle=":",linewidth=1.5,color="#2ca02c",label="q90 best-case stability")
ax1.plot([2026,2027],[pubs[-1],mean5[0]],color="#d62728",linewidth=2.5,linestyle="--",alpha=0.6)
ax1.axvline(2024,color="gray",linestyle="-.",linewidth=1.2,alpha=0.7)
ymin,ymax=ax1.get_ylim()
ax1.text(2024.05, pubs.max()*0.55,"AI automation\ninflection (2024)",fontsize=9,color="gray",va="center")
ax1.set_xlabel("Year",fontsize=12); ax1.set_ylabel("Meta-Analysis Publications per Year",fontsize=12)
ax1.set_title("Meta-Analysis Publications: History (2015–2026) vs TimesFM-1.0-200M Forecast (2027–2031)\nZero-shot foundation forecast under AI-automation structural shift",fontsize=12,pad=12)
ax1.grid(True,alpha=0.3); ax1.set_xticks(list(years_hist)+FORECAST_YEARS); ax1.tick_params(axis="x",rotation=45)
ax2=ax1.twinx()
ai_all=list(ai_hist)+[0.92,0.95,0.97,0.98,0.99]
ax2.plot(list(years_hist)+FORECAST_YEARS,ai_all,color="purple",linewidth=1.8,linestyle="-.",marker="x",markersize=5,alpha=0.75,label="AI Automation Impact Factor (hist + projected)")
ax2.set_ylabel("AI Automation Impact Factor (0–1)",fontsize=11,color="purple")
ax2.tick_params(axis="y",labelcolor="purple"); ax2.set_ylim(-0.05,1.05)
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax1.legend(h1+h2,l1+l2,loc="upper left",fontsize=9,framealpha=0.95)
fig.tight_layout()
out_png="/home/user/meta_analysis_forecast_2027_2031.png"
fig.savefig(out_png,dpi=150,bbox_inches="tight")
flush_print(f"Saved {out_png}")
plt.close(fig)
log_mem("final")
flush_print("DONE")
