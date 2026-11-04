---
name: mlk-datetime-features
description: Parse date strings and build ML-ready temporal features in pure Python (pandas datetime + stdlib zoneinfo) - to_datetime with an explicit format and errors="coerce", tz_localize/tz_convert across time zones, splitting a datetime column into year/month/day/hour/minute and weekday (name + Monday=0), and timedelta durations between two datetime columns. Use when converting string dates, adding or changing time zones, breaking a timestamp into component features, encoding day-of-week, or computing the number of days between two timestamps.
---

# Datetime Features for ML (Python)

## Purpose
Turn raw date/time strings into a `datetime64` column, then extract the numeric/label temporal
features a model can consume (components, weekday, duration) and carry correct time-zone info
(Albon ch.7 p124-135). Parsing uses pandas' `to_datetime`, which centralises many other libraries
(Albon p124).

## Inputs
- A 1-D array/list/Series of date strings, or an existing `datetime64` column.
- A `format` string built from the codes below; an optional IANA `tz` name (e.g. `Europe/London`).
- For durations: two `datetime64` columns (start, end). Demo data is Albon's own inline examples.

## Algorithms (all pure numpy/scipy)

### Converting strings to datetimes (Albon p124-125)
```python
import numpy as np, pandas as pd
date_strings = np.array(['03-04-2005 11:35 PM', '23-05-2010 12:01 AM', '04-09-2009 09:09 PM'])
parsed = [pd.to_datetime(s, format='%d-%m-%Y %I:%M %p', errors='coerce') for s in date_strings]
print(f"RESULT parsed={[str(x) for x in parsed]}")
```
`errors="coerce"` sets an unparsable value to `NaT` instead of raising (Albon p125). Prints
`['2005-04-03 23:35:00', '2010-05-23 00:01:00', '2009-09-04 21:09:00']` (Albon p125).

### Adding and converting time zones (Albon p126-127)
```python
import pandas as pd, zoneinfo
london = pd.Timestamp('2017-05-01 06:00:00').tz_localize('Europe/London')   # naive -> +0100 (BST)
abidjan = london.tz_convert('Africa/Abidjan')                               # same instant -> 05:00
series = pd.Series(pd.date_range('2/2/2002', periods=3, freq='ME'))
localized = series.dt.tz_localize('Africa/Abidjan')                          # .dt applies to every element
print(f"RESULT london={london} abidjan={abidjan} tz={sorted(zoneinfo.available_timezones())[:2]}")
```
Prints `london=2017-05-01 06:00:00+01:00`, `abidjan=2017-05-01 05:00:00+00:00`,
`tz=['Africa/Abidjan', 'Africa/Accra']` (Albon p126-127).

### Breaking a datetime into component + weekday features (Albon p128-131)
```python
import pandas as pd
df = pd.DataFrame({'date': pd.date_range('1/1/2001', periods=150, freq='W')})
for part in ('year', 'month', 'day', 'hour', 'minute'):
    df[part] = getattr(df['date'].dt, part)
wk = pd.Series(pd.date_range('2/2/2002', periods=3, freq='ME'))
print(f"RESULT row0={df.year[0]}-{df.month[0]}-{df.day[0]} {df.hour[0]:02d}:{df.minute[0]:02d} "
      f"weekday={list(wk.dt.day_of_week)} names={list(wk.dt.day_name())}")
```
`dt` exposes the time properties (Albon p128). `day_of_week` is Monday=0 (Albon p130-131). Row0 is
`2001-1-7 00:00`, weekday `[3, 6, 1]`, names `['Thursday', 'Sunday', 'Tuesday']` (Albon p129, p130).

### Duration between two datetime columns (Albon p129-130)
```python
import pandas as pd
df = pd.DataFrame({'Arrived': [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')],
                   'Left':    [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]})
stay_days = (df['Left'] - df['Arrived']).dt.days           # timedelta64[ns] -> integer day count
print(f"RESULT stay_days={list(stay_days)} dtype={stay_days.dtype}")
```
Subtracting the two features yields a `TimeDelta` (Albon p129-130). Prints `stay_days=[0, 2]`.

## Output contract
Emit `datetime64[ns]` (optionally `...[tz]`) columns plus derived feature columns:
`year, month, day, hour, minute` (int64), `day_of_week` (int64, Monday=0) and/or `day_name`
(object), and any duration as an integer count (`.dt.days` / `.dt.total_seconds()`). Unparsable
strings become `NaT`. Print one `RESULT name=value` line per quantity.

## Parsing rules
- Always pass an explicit `format`; the same calendar date appears as `03-23-15` in one source and
  `3|23|2015` in another, so guessing mis-parses day vs month (Albon p125).
- Format codes (Albon p125): `%Y` full year, `%m` month 0-padded, `%d` day 0-padded,
  `%I` hour on the 12-h clock 0-padded, `%p` AM/PM, `%M` minute, `%S` second.
- 12-hour clock strings need both `%I` and `%p`; omitting `%p` mis-locates PM times.

## Validity conditions
- A timestamp with no zone is naive; `tz_localize` before comparing across sources or the offset
  is undefined (Albon p126).
- `tz_convert` relabels the same instant, it does not shift it - the wall clock changes but the
  moment does not (Albon p126).
- Weekday numbers are only comparable if every source uses the Monday=0 convention (Albon p130-131).
- A duration is only meaningful when start <= end; negative `stay_days` flags a data-integrity bug.

## Conventions
- Libs: pandas datetime for all parsing/tz/features; stdlib `zoneinfo` for the timezone-name list.
- The book imports `pytz.all_timezones` (Albon p127); **pytz is not installed**, so the same list is
  reimplemented with stdlib `zoneinfo.available_timezones()` (no numeric change). `dateutil` is present.
- Legacy freq aliases `M`/`H` are removed on this stack: use `ME` (month-end) and `h` (hour) instead.
- Use Albon's own inline example values for the worked numbers; add `numpy.random.default_rng(42)`
  only when a block needs synthetic rows. Never hardcode a result - compute and print `RESULT`.

## Out of scope
Cyclic (sin/cos) encoding of hour/day-of-year is named in the capability but is NOT in the grounded
ch.7 pages, so it is left out rather than invented (Albon ch.7 p124-135). Lagged features via `shift`
(7.7), rolling/moving windows (7.8) and time-series `interpolate`/`ffill`/`bfill` (7.9) belong to the
time-series feature and resampling skills, not here.
