---
name: pda-time-series
description: Date/timestamp APIs, date_range and frequency aliases, shifting, resampling (down/up-sample, ohlc), rolling/expanding windows with min_periods, EWMA (span->alpha), timezone localize/convert across DST, and Period/asfreq conversions, in pure pandas/numpy (pandas datetime, resample, rolling). Use when asked to resample a time series, build a moving/EWMA average, parse dates, handle time zones, or do period/frequency arithmetic.
---

# Time Series with pandas (Python)

## Purpose
Reshape and summarise a timestamp- or period-indexed series: parse dates, fix a frequency, resample,
smooth with moving/expanding/EWMA windows, convert time zones, and do period arithmetic. A time series is
a Series/DataFrame indexed by `DatetimeIndex`/`PeriodIndex`; every op aligns on the index, excluding NA (McKinney p375-380).

## Inputs
- `series`: `pd.Series`/`pd.DataFrame` indexed by `pd.DatetimeIndex` or `pd.PeriodIndex` (index must be the time axis).
- `rule`: frequency string, e.g. `"D"`, `"min"`, `"5min"`, `"ME"` (month-end), `"BME"` (business month-end), `"Y-DEC"` (McKinney p386-387).
- `window`: number of periods for `rolling`, or a time offset string like `"20D"`; `span` for `ewm`.
- Seeded demo data via `numpy.random.default_rng(42)`; the book's stock/macrodata downloads are not used.

## Algorithms (all pure numpy/scipy)

### Timestamps, NaT and duplicate-index aggregation
```python
import numpy as np, pandas as pd
ts = pd.Series(np.array([-0.204708,0.478943,-0.519439,-0.555730,1.965781,1.393406]),
               index=pd.to_datetime(["2011-01-02","2011-01-05","2011-01-07","2011-01-08","2011-01-10","2011-01-12"]))
assert str(ts.index.dtype).startswith("datetime64")                 # vectorised ns/us timestamps (McKinney p380)
assert pd.isna(pd.to_datetime(["2011-07-06", None]))[1]             # NaT is the null timestamp (McKinney p379)
rng = ts["2011-01-06":"2011-01-11"]                                 # range query with absent endpoints (McKinney p382)
assert list(rng.index.strftime("%Y-%m-%d")) == ["2011-01-07","2011-01-08","2011-01-10"]
dup = pd.Series(np.arange(5), index=pd.DatetimeIndex(
    ["2000-01-01","2000-01-02","2000-01-02","2000-01-02","2000-01-03"]))
assert not dup.index.is_unique
g = dup.groupby(level=0)                                            # aggregate non-unique timestamps (McKinney p384)
assert g.mean()["2000-01-02"] == 2.0 and g.count()["2000-01-02"] == 3
print(f"RESULT range_query_len={len(rng)} dup_mean={g.mean()['2000-01-02']} dup_count={g.count()['2000-01-02']}")
```
Prints `dup_mean=2.0 dup_count=3` - the book's duplicate-index example (McKinney p384).

### Date ranges, frequencies, shifting and anchored offsets
```python
import numpy as np, pandas as pd
from datetime import datetime
from pandas.tseries.offsets import MonthEnd
bm = pd.date_range("2000-01-01","2000-12-01",freq="BME")            # last business day of each month (McKinney p386)
assert len(bm) == 11 and str(bm[-1].date()) == "2000-11-30"
ts = pd.Series(np.arange(1.,5.), index=pd.date_range("2000-01-31",periods=4,freq="ME"))
pc = ts/ts.shift(1)-1                                               # consecutive percent change (McKinney p390)
assert np.isnan(pc.iloc[0]) and abs(pc.iloc[1]-1.0) < 1e-12
now = datetime(2011,11,17)
assert str(MonthEnd().rollforward(now)) == "2011-11-30 00:00:00"    # anchored offset rolls forward (McKinney p391)
assert str(MonthEnd().rollback(now)) == "2011-10-31 00:00:00"
print(f"RESULT bm_n={len(bm)} bm_last={bm[-1].date()} pct1={pc.iloc[1]}")
```
`bm` has 11 dates ending `2000-11-30`; the naive `shift` leaves the index fixed, `shift(n,freq=...)` advances the stamps (McKinney p389-391).

### Resampling: downsampling, ohlc and upsampling
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)                                     # seeded, offline, reproducible
dates = pd.date_range("2000-01-01", periods=100)
ts = pd.Series(rng.standard_normal(100), index=dates)
monthly = ts.resample("ME").mean()                                  # daily -> month-end buckets (McKinney p405-406)
assert len(monthly) == 4 and str(monthly.index[-1].date()) == "2000-04-30"
mn = pd.Series(np.arange(12), index=pd.date_range("2000-01-01",periods=12,freq="min"))
assert list(mn.resample("5min").sum()) == [10,35,21]                # default closed=left (McKinney p407)
assert list(mn.resample("5min",closed="right",label="right").sum()) == [0,15,40,11]   # (McKinney p408)
assert list(mn.resample("5min").ohlc().iloc[0]) == [0,4,0,4]        # open/high/low/close per bin (McKinney p409)
assert mn[:3].to_frame("v").resample("min").asfreq().notna().all().all()   # upsample without aggregation (McKinney p410)
print(f"RESULT monthly_n={len(monthly)} monthly_last={round(monthly.iloc[-1],4)} sum5min={list(mn.resample('5min').sum())}")
```
The 5-minute sums `[10,35,21]` reproduce the book's hand-checkable downsample (McKinney p407).

### Rolling and expanding windows
```python
import numpy as np, pandas as pd
assert list(pd.Series([1,2,3,4,5]).rolling(3).mean().dropna()) == [2.0,3.0,4.0]     # equal-weight sliding window (McKinney p414)
mp = pd.Series([1,2,3,4,5]).rolling(3, min_periods=2).mean()                         # tolerate short head (McKinney p415)
assert np.isnan(mp.iloc[0]) and mp.iloc[1] == 1.5
exp = list(pd.Series([1,2,3,4]).expanding().mean())                                  # window grows to full series (McKinney p416)
assert exp == [1.0, 1.5, 2.0, 2.5]
t = pd.Series(np.arange(1.,7.), index=pd.date_range("2000-01-01", periods=6))
assert list(t.rolling("3D").mean()) == [1.0,1.5,2.0,3.0,4.0,5.0]                     # fixed time-offset window (McKinney p417)
print(f"RESULT roll3={[2.0,3.0,4.0]} min_periods_first={mp.iloc[1]} expanding={exp}")
```
`rolling(N).mean()` averages the trailing N periods; a string window like `"20D"` is essential for irregular data (McKinney p417).

### Exponentially weighted moving average (EWMA)
```python
import numpy as np, pandas as pd
x = pd.Series([1.0, 2.0, 3.0])
alpha = 2/(3+1)                                                     # span N -> alpha = 2/(N+1) (McKinney p417-418)
ew = x.ewm(span=3, adjust=False).mean()
assert abs(alpha - 0.5) < 1e-12
assert list(ew) == [1.0, 1.5, 2.25]                                # y_t=(1-a)*y_{t-1}+a*x_t, y_0=x_0
assert bool((x.ewm(span=1, adjust=False).mean() == x).all())       # span 1 -> alpha 1 -> identity
print(f"RESULT ewm_span3={[round(v,4) for v in ew]} alpha={alpha}")
```
EWMA weights recent observations more and adapts faster than an equal-weight window; span ties it to an equivalent window length (McKinney p417-418).

### Time zones: localize, convert, DST, instant invariance
```python
import numpy as np, pandas as pd
dates = pd.date_range("2012-03-09 09:30", periods=6)
ts = pd.Series(np.arange(6.), index=dates)
assert ts.index.tz is None                                          # naive by default (McKinney p393)
utc = ts.tz_localize("UTC")
ist = utc.tz_convert("Asia/Kolkata")
assert str(ist.index[0]) == "2012-03-09 15:00:00+05:30"            # 09:30 + fixed +05:30 (McKinney p394-395)
ny = utc.tz_convert("America/New_York")
assert all(ny.index[k] == utc.index[k] for k in range(len(ny)))    # conversion never changes the instant
st = pd.Timestamp("2011-03-12 04:00").tz_localize("UTC")
assert st.value == st.tz_convert("America/New_York").value         # UTC ns-since-epoch preserved (McKinney p396)
print(f"RESULT ist0={ist.index[0]} utc_value={st.value}")
```
`utc_value=1299902400000000000` matches the book; New York offsets sit at -05:00/-04:00 across the DST boundary (McKinney p394-396).

### Periods and frequency conversion (asfreq)
```python
import pandas as pd
p = pd.Period("2011", freq="Y-DEC")                                 # whole-year span (McKinney p397-398)
assert str(p+5) == "2016" and str(p-2) == "2009"
assert (pd.Period("2014", freq="Y-DEC") - p).n == 3                # difference is a 3-year offset (McKinney p398)
assert str(p.asfreq("M", how="start")) == "2011-01" and str(p.asfreq("M", how="end")) == "2011-12"  # (McKinney p399)
q = pd.Period("2012Q4", freq="Q-JAN")
assert str(q.asfreq("D", how="start")) == "2011-11-01" and str(q.asfreq("D", how="end")) == "2012-01-31"  # FY ends Jan (McKinney p401)
p4pm = (q.asfreq("B", how="end") - 1).asfreq("min", how="start") + 16*60   # 2nd-to-last business day, 4 p.m. (McKinney p401)
assert str(p4pm) == "2012-01-30 16:00"
print(f"RESULT q_start={q.asfreq('D',how='start')} p4pm={p4pm} p_plus5={p+5}")
```
A period's meaning depends on the fiscal anchor: `2012Q4` under `Q-JAN` spans 2011-11-01 to 2012-01-31 (McKinney p401).

## Output contract
Every block prints `RESULT name=value` lines. Resample/rolling/ewm return a `Series`/`DataFrame` on the new
frequency (labelled per `label`/`closed`); timezone methods return an index with a non-`None` `.tz`; period
methods return `Period`/`PeriodIndex`. Report the frequency alias and bin-edge convention with each aggregate.

## Validity conditions
- Resample/rolling require a monotonic time index; irregular data must use a time-offset window (`"20D"`) not a period count (McKinney p417).
- `min_periods` defaults to the window size, so the first `window-1` values are NaN unless lowered (McKinney p415).
- Downsampling/upsampling of periods requires the target to be a sub/super-period, else pandas raises (McKinney p412).
- Naive and time-zone-aware data cannot be combined; localizing ambiguous/non-existent DST times needs an explicit rule (McKinney p396-397).

## Reporting conventions
State the frequency alias and the bin-edge convention (`closed`/`label`) next to every resampled number
and give units on the timestamp; round only the reported aggregate, keep full precision internally.

## Workflow
Parse/normalise the index -> assert fixed or irregular frequency -> resample or shift to the target grid ->
smooth (rolling/expanding/ewm) -> convert time zones/periods -> aggregate and print `RESULT` lines.

## Out of scope
Model fitting on time series (AR/statsmodels), calendar-logic beyond pandas built-ins, and elapsed-time/
timedelta indexes are out of scope here; this skill covers the datetime/resample/window API only.

## Conventions
- Freq aliases follow installed pandas 3.x: `ME`/`Y` for month/year-end, `min`/`s` for minute/second; the
  book's older `M`/`A`/`T`/`H` aliases are renamed accordingly. No package is missing on the numpy/pandas stack.
- RNG: `numpy.random.default_rng(42)`; never hardcode a number - compute it and cite the method page.
- Timestamps store at `datetime64` ns/us resolution; do not assume `.index.dtype == datetime64[ns]` under pandas 3.
- Source: McKinney, *Python for Data Analysis*, 3rd ed., Ch.11 "Time Series" (PDF pages 375-421).
